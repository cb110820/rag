// ── 全局状态 ──────────────────────────────────────────────────────────
const API = ""; // 同源部署，留空
let token = localStorage.getItem("rag_token") || "";
let userEmail = localStorage.getItem("rag_email") || "";
let userRole = localStorage.getItem("rag_role") || "";
let currentSlug = null;

// ── 工具函数 ──────────────────────────────────────────────────────────
function toast(msg, type = "info") {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = `toast ${type}`;
  el.classList.remove("hidden");
  setTimeout(() => el.classList.add("hidden"), 2500);
}

async function api(method, path, body, isForm = false) {
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!isForm && body) headers["Content-Type"] = "application/json";
  const opts = { method, headers };
  if (body) opts.body = isForm ? body : JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (res.status === 401) { logout(); throw new Error("未登录"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `请求失败 (${res.status})`);
  return data;
}

function fmtTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

// ── 认证 ──────────────────────────────────────────────────────────────
function showApp() {
  document.getElementById("auth-layer").classList.add("hidden");
  document.getElementById("app").classList.remove("hidden");
  document.getElementById("user-email").textContent = userEmail;
  // 平台管理员显示「管理」标签页（全局技能上传）
  document.getElementById("nav-tab-admin").classList.toggle("hidden", userRole !== "platform_admin");
  // 登录后默认落在对话页：加载历史会话并进入可直接输入的新对话状态
  loadThreadList();
  startNewThread();
}

function logout() {
  token = "";
  userEmail = "";
  userRole = "";
  localStorage.removeItem("rag_token");
  localStorage.removeItem("rag_email");
  localStorage.removeItem("rag_role");
  document.getElementById("auth-layer").classList.remove("hidden");
  document.getElementById("app").classList.add("hidden");
}

document.getElementById("auth-login-btn").addEventListener("click", async () => {
  const email = document.getElementById("auth-email").value.trim();
  const password = document.getElementById("auth-password").value;
  const msg = document.getElementById("auth-msg");
  if (!email || !password) { msg.textContent = "请输入邮箱和密码"; return; }
  try {
    const data = await api("POST", "/auth/login", { email, password });
    token = data.access_token;
    userEmail = email;
    userRole = (data.user && data.user.role) || "";
    localStorage.setItem("rag_token", token);
    localStorage.setItem("rag_email", email);
    localStorage.setItem("rag_role", userRole);
    msg.textContent = "";
    showApp();
    toast("登录成功", "success");
  } catch (e) {
    msg.textContent = e.message;
  }
});

document.getElementById("logout-btn").addEventListener("click", logout);

// ── 导航 ──────────────────────────────────────────────────────────────
document.querySelectorAll(".nav-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const view = tab.dataset.view;
    document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
    document.getElementById(`view-${view}`).classList.remove("hidden");
    if (view === "chat") loadThreadList();
    if (view === "wiki") loadWikiList();
    if (view === "knowledge") loadKbFiles();
    if (view === "admin") loadAdminSkills();
  });
});

// ── Wiki ──────────────────────────────────────────────────────────────
let allWikiPages = [];

// ── 页面类型徽标 + 矛盾待裁决检测（WeKnora 化） ────────────────────────
const _wikiPageCache = {}; // slug → 最近一次拉取的正文

function kindOf(p) {
  if (p.slug === "index") return "index";
  if (p.slug === "log") return "log";
  if (p.slug.startsWith("entity-")) return "entity";
  if (p.slug.startsWith("doc-")) return "doc";
  // 人工页：从缓存正文里识别 frontmatter kind，兜底 page
  const fm = (_wikiPageCache[p.slug] || "").match(/^---\n([\s\S]*?)\n---/);
  if (fm) {
    const m = fm[1].match(/^kind:\s*["']?(\w+)/m);
    if (m && ["doc", "entity", "index", "log"].includes(m[1])) return m[1];
  }
  return "page";
}

const _KIND_LABELS = { doc: "文档", entity: "实体", index: "目录", log: "日志", page: "页面" };

function kindBadge(kind) {
  return `<span class="wiki-kind-badge k-${kind}">${_KIND_LABELS[kind] || "页面"}</span>`;
}

function countContradictions(content) {
  const open = (content.match(/<!--\s*contradiction:open\s*-->/g) || []).length;
  const resolved = (content.match(/<!--\s*contradiction:resolved\s*-->/g) || []).length;
  return { open, resolved, has: open > resolved };
}

// 并发限流预取列表页正文，找出含未裁决矛盾的页面
async function probeContradictions(pages) {
  const targets = pages.filter(p => _wikiPageCache[p.slug] === undefined);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(6, targets.length) }, async () => {
    while (cursor < targets.length) {
      const p = targets[cursor++];
      try {
        const page = await api("GET", `/wiki/pages/${p.slug}`);
        _wikiPageCache[p.slug] = page.content || "";
      } catch { _wikiPageCache[p.slug] = ""; }
    }
  });
  await Promise.allSettled(workers);
  return pages
    .filter(p => countContradictions(_wikiPageCache[p.slug] || "").has)
    .map(p => p.slug);
}

async function loadWikiList() {
  const kw = document.getElementById("wiki-search-input").value.trim();
  // 有搜索词走全文搜索端点，无词走普通列表
  if (kw) { await runWikiSearch(kw); return; }
  try {
    allWikiPages = await api("GET", "/wiki/pages");
    renderWikiList(allWikiPages);
    // 异步补全矛盾标记：预取完成后带警示徽标重渲染
    probeContradictions(allWikiPages).then(slugSet => {
      renderWikiList(allWikiPages, new Set(slugSet));
    });
  } catch (e) { toast(e.message, "error"); }
}

function renderWikiList(pages, warnSlugs) {
  warnSlugs = warnSlugs || new Set();
  const ul = document.getElementById("wiki-list");
  ul.innerHTML = "";
  if (pages.length === 0) {
    ul.innerHTML = '<li class="wiki-search-empty">暂无页面</li>';
    return;
  }
  pages.forEach(p => {
    const li = document.createElement("li");
    const tag = p.title_match ? '<span class="wiki-title-match-tag">标题命中</span>' : "";
    const badge = kindBadge(kindOf(p));
    const warn = warnSlugs.has(p.slug)
      ? '<span class="wiki-warn-badge" title="存在待裁决矛盾，结论可能互相冲突">⚠ 矛盾</span>' : "";
    const snippet = p.snippet
      ? `<div class="wiki-snippet">${highlight(p.snippet, currentSearchKw)}</div>` : "";
    li.innerHTML = `<div class="wiki-title">${badge}${escapeHtml(p.title)}${tag}${warn}</div><div class="wiki-slug">${p.slug}</div>${snippet}`;
    li.addEventListener("click", () => {
      document.getElementById("wiki-search-input").value = "";
      currentSearchKw = "";
      loadWikiPage(p.slug);
    });
    ul.appendChild(li);
  });
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function highlight(text, kw) {
  const safe = escapeHtml(text);
  if (!kw) return safe;
  const re = new RegExp(`(${kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
  return safe.replace(re, "<mark>$1</mark>");
}

// ── 全文搜索（防抖） ─────────────────────────────────────────────────
let currentSearchKw = "";
let searchTimer = null;

async function runWikiSearch(kw) {
  currentSearchKw = kw;
  try {
    const hits = await api("GET", `/wiki/search?q=${encodeURIComponent(kw)}`);
    // 搜索过程中关键词已被清空则丢弃过期结果
    if (document.getElementById("wiki-search-input").value.trim() !== kw) return;
    const ul = document.getElementById("wiki-list");
    ul.innerHTML = "";
    if (hits.length === 0) {
      ul.innerHTML = `<li class="wiki-search-empty">未找到包含「${escapeHtml(kw)}」的页面</li>`;
      return;
    }
    renderWikiList(hits);
  } catch (e) { toast(e.message, "error"); }
}

document.getElementById("wiki-search-input").addEventListener("input", e => {
  clearTimeout(searchTimer);
  const kw = e.target.value.trim();
  if (!kw) { loadWikiList(); return; }
  searchTimer = setTimeout(() => runWikiSearch(kw), 300);
});

async function loadWikiPage(slug) {
  currentSlug = slug;
  document.querySelectorAll("#wiki-list li").forEach((li, i) => {
    li.classList.toggle("active", li.querySelector(".wiki-slug")?.textContent === slug);
  });
  try {
    const page = await api("GET", `/wiki/pages/${slug}`);
    _wikiPageCache[slug] = page.content || "";
    document.getElementById("wiki-editor-empty").classList.add("hidden");
    document.getElementById("wiki-editor-form").classList.remove("hidden");
    document.getElementById("wiki-slug").value = page.slug;
    document.getElementById("wiki-slug").disabled = true;
    document.getElementById("wiki-title").value = page.title;
    document.getElementById("wiki-content").value = page.content;
    document.getElementById("wiki-history").classList.add("hidden");
    renderContradictionBanner(page.content);
    renderWikiIndexNav(page.content);
    renderWikiLogView(page.content);
    renderPreview();
    loadWikiAttachments(slug);
  } catch (e) { toast(e.message, "error"); }
}

// ── 矛盾待裁决横幅 / index 目录导航 / log 沉淀日志视图 ─────────────────

function renderContradictionBanner(content) {
  const banner = document.getElementById("wiki-contradiction-banner");
  if (!banner) return;
  const c = countContradictions(content || "");
  if (c.has) {
    banner.textContent =
      `⚠ 此页面存在 ${c.open} 处待裁决矛盾（自动沉淀发现新旧结论冲突）。` +
      "下方结论可能互相冲突，请人工核对后编辑标记 contradiction:resolved。";
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

// index 页：把 `## 分区` 与 `- [标题](slug.md)` 渲染为可点击目录面板
function renderWikiIndexNav(content) {
  const nav = document.getElementById("wiki-index-nav");
  if (!nav) return;
  if (currentSlug !== "index" || !content) { nav.classList.add("hidden"); nav.innerHTML = ""; return; }
  let section = "其他";
  const items = [];
  content.split("\n").forEach(line => {
    const h = line.match(/^##\s+(.+)/);
    if (h) { section = h[1].trim(); return; }
    const li = line.match(/^-\s+\[([^\]]+)\]\(([^)]+?)(?:\.md)?\)/);
    if (li && !li[2].startsWith("http")) {
      items.push({ section, title: li[1], slug: li[2] });
    }
  });
  if (items.length === 0) { nav.classList.add("hidden"); return; }
  const bySection = {};
  items.forEach(it => (bySection[it.section] = bySection[it.section] || []).push(it));
  nav.innerHTML = Object.entries(bySection).map(([sec, list]) => `
    <div class="wiki-index-nav-section">
      <div class="wiki-index-nav-heading">${escapeHtml(sec)}</div>
      <div class="wiki-index-nav-links">
        ${list.map(it =>
          `<a href="#" data-slug="${escapeHtml(it.slug)}" class="wiki-index-nav-link">${escapeHtml(it.title)}</a>`
        ).join("")}
      </div>
    </div>`).join("");
  nav.classList.remove("hidden");
  nav.querySelectorAll(".wiki-index-nav-link").forEach(a => {
    a.addEventListener("click", ev => {
      ev.preventDefault();
      loadWikiPage(a.dataset.slug);
    });
  });
}

// log 页：把 `- 时间 \`source\` 沉淀成功/失败：...` 渲染为双色日志列表
function renderWikiLogView(content) {
  const box = document.getElementById("wiki-log-view");
  if (!box) return;
  if (currentSlug !== "log" || !content) { box.classList.add("hidden"); box.innerHTML = ""; return; }
  const re = /^-\s+(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+`([^`]+)`\s+沉淀(成功|失败)[：:]\s*(.*)$/gm;
  const entries = [];
  let m;
  while ((m = re.exec(content)) !== null) {
    entries.push({ ts: m[1], source: m[2], ok: m[3] === "成功", detail: m[4] });
  }
  if (entries.length === 0) { box.classList.add("hidden"); return; }
  box.innerHTML = `<div class="wiki-log-heading">沉淀记录（${entries.length} 条）</div>` + entries.map(e => `
    <div class="wiki-log-item ${e.ok ? "ok" : "fail"}">
      <span class="wiki-log-status">${e.ok ? "✓ 成功" : "✗ 失败"}</span>
      <span class="wiki-log-source">${escapeHtml(e.source)}</span>
      <span class="wiki-log-time">${escapeHtml(e.ts)}</span>
      <div class="wiki-log-detail">${escapeHtml(e.detail)}</div>
    </div>`).join("");
  box.classList.remove("hidden");
}

document.getElementById("wiki-new-btn").addEventListener("click", () => {
  currentSlug = null;
  document.getElementById("wiki-editor-empty").classList.add("hidden");
  ["wiki-contradiction-banner", "wiki-index-nav", "wiki-log-view"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.add("hidden"); el.innerHTML = ""; }
  });
  document.getElementById("wiki-editor-form").classList.remove("hidden");
  document.getElementById("wiki-slug").value = "";
  document.getElementById("wiki-slug").disabled = false;
  document.getElementById("wiki-title").value = "";
  document.getElementById("wiki-content").value = "";
  document.getElementById("wiki-summary").value = "";
  document.getElementById("wiki-history").classList.add("hidden");
  document.getElementById("wiki-attachments").classList.add("hidden");
  document.getElementById("wiki-attachment-list").innerHTML = "";
  document.querySelectorAll("#wiki-list li").forEach(li => li.classList.remove("active"));
  renderPreview();
});

// ── Markdown 实时预览 + 编辑模式切换 ─────────────────────────────────
function renderPreview() {
  const md = document.getElementById("wiki-content").value;
  const box = document.getElementById("wiki-preview");
  if (window.marked && typeof window.marked.parse === "function") {
    box.innerHTML = window.marked.parse(md || "");
  } else {
    // CDN 加载失败时降级为纯文本
    box.textContent = md || "";
  }
}

// 输入即渲染（防抖 150ms，避免大文档频繁解析）；矛盾横幅同步刷新
let previewTimer = null;
document.getElementById("wiki-content").addEventListener("input", () => {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => {
    renderPreview();
    renderContradictionBanner(document.getElementById("wiki-content").value);
  }, 150);
});

// 分屏 / 仅编辑 / 仅预览 切换
document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const mode = btn.dataset.mode;
    const wrap = document.getElementById("wiki-content-wrap");
    wrap.className = `wiki-content-wrap mode-${mode}`;
    renderPreview();
  });
});

// ── Markdown 工具栏 ───────────────────────────────────────────────────
const MD_TOOLS = [
  { key: "h1", label: "H1", title: "一级标题" },
  { key: "h2", label: "H2", title: "二级标题" },
  { key: "h3", label: "H3", title: "三级标题" },
  { sep: true },
  { key: "bold", label: "B", title: "加粗", cls: "tb-bold" },
  { key: "italic", label: "I", title: "斜体", cls: "tb-italic" },
  { key: "strike", label: "S", title: "删除线", cls: "tb-strike" },
  { sep: true },
  { key: "link", label: "🔗", title: "插入链接" },
  { key: "image", label: "🖼", title: "插入图片" },
  { key: "code", label: "</>", title: "行内代码", cls: "tb-code" },
  { key: "codeblock", label: "{}", title: "代码块", cls: "tb-code" },
  { sep: true },
  { key: "quote", label: "❝", title: "引用" },
  { key: "ul", label: "• 列表", title: "无序列表" },
  { key: "ol", label: "1. 列表", title: "有序列表" },
  { key: "table", label: "表格", title: "插入表格" },
  { key: "hr", label: "—", title: "分隔线" },
];

const mdToolbar = document.getElementById("md-toolbar");
MD_TOOLS.forEach(t => {
  if (t.sep) {
    const sep = document.createElement("span");
    sep.className = "md-sep";
    mdToolbar.appendChild(sep);
    return;
  }
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `md-tool ${t.cls || ""}`;
  btn.title = t.title;
  btn.textContent = t.label;
  btn.addEventListener("click", () => applyMdTool(t.key));
  mdToolbar.appendChild(btn);
});

// 行内包裹：**文字** / *文字* / [文字](url) 等
function mdWrap(before, after, placeholder) {
  const ta = document.getElementById("wiki-content");
  const s = ta.selectionStart, e = ta.selectionEnd;
  const sel = ta.value.slice(s, e) || placeholder;
  ta.focus();
  ta.setRangeText(before + sel + after, s, e, "select");
  // setRangeText("select") 选中整个插入内容，进一步把选区收窄到中间文本
  ta.setSelectionRange(s + before.length, s + before.length + sel.length);
  ta.dispatchEvent(new Event("input"));
}

// 行前缀（标题/引用/列表）：作用于选区覆盖的每一行；已全部带前缀则取消
function mdPrefix(prefixFn) {
  const ta = document.getElementById("wiki-content");
  const s = ta.selectionStart, e = ta.selectionEnd;
  const text = ta.value;
  const lineStart = text.lastIndexOf("\n", s - 1) + 1;
  let lineEnd = text.indexOf("\n", e);
  if (lineEnd === -1) lineEnd = text.length;
  const lines = text.slice(lineStart, lineEnd).split("\n");
  const newLines = lines.map((line, i) => {
    const prefix = typeof prefixFn === "function" ? prefixFn(i, line) : prefixFn;
    return line.startsWith(prefix) ? line.slice(prefix.length) : prefix + line;
  });
  ta.focus();
  ta.setRangeText(newLines.join("\n"), lineStart, lineEnd, "select");
  ta.dispatchEvent(new Event("input"));
}

// 插入块级元素（代码块/表格/分隔线），自动补前置空行
function mdBlock(block) {
  const ta = document.getElementById("wiki-content");
  const s = ta.selectionStart;
  const before = ta.value.slice(0, s);
  const lead = before.length === 0 || before.endsWith("\n\n") ? ""
    : before.endsWith("\n") ? "\n" : "\n\n";
  ta.focus();
  ta.setRangeText(lead + block + "\n", s, ta.selectionEnd, "end");
  ta.dispatchEvent(new Event("input"));
}

function applyMdTool(key) {
  switch (key) {
    case "h1": mdPrefix("# "); break;
    case "h2": mdPrefix("## "); break;
    case "h3": mdPrefix("### "); break;
    case "bold": mdWrap("**", "**", "加粗文本"); break;
    case "italic": mdWrap("*", "*", "斜体文本"); break;
    case "strike": mdWrap("~~", "~~", "删除文本"); break;
    case "code": mdWrap("`", "`", "code"); break;
    case "link": mdWrap("[", "](https://)", "链接文本"); break;
    case "image": mdWrap("![", "](图片地址)", "图片描述"); break;
    case "quote": mdPrefix("> "); break;
    case "ul": mdPrefix("- "); break;
    case "ol": mdPrefix((i) => `${i + 1}. `); break;
    case "hr": mdBlock("\n---\n"); break;
    case "codeblock":
      mdBlock("```text\n代码片段\n```\n");
      break;
    case "table":
      mdBlock("| 列 1 | 列 2 | 列 3 |\n| --- | --- | --- |\n| 内容 | 内容 | 内容 |\n");
      break;
  }
}

document.getElementById("wiki-save-btn").addEventListener("click", async () => {
  const slug = document.getElementById("wiki-slug").value.trim();
  const title = document.getElementById("wiki-title").value.trim();
  const content = document.getElementById("wiki-content").value;
  const summary = document.getElementById("wiki-summary").value.trim();
  if (!slug || !title) { toast("Slug 和标题必填", "error"); return; }
  try {
    if (currentSlug) {
      await api("PUT", `/wiki/pages/${currentSlug}`, { title, content, summary });
      toast("更新成功", "success");
    } else {
      await api("POST", "/wiki/pages", { slug, title, content, summary });
      currentSlug = slug;
      toast("创建成功", "success");
    }
    loadWikiList();
  } catch (e) { toast(e.message, "error"); }
});

document.getElementById("wiki-delete-btn").addEventListener("click", async () => {
  if (!currentSlug) return;
  if (!confirm(`确定删除页面 "${currentSlug}" 吗？此操作不可撤销。`)) return;
  try {
    await api("DELETE", `/wiki/pages/${currentSlug}`);
    toast("已删除", "success");
    currentSlug = null;
    document.getElementById("wiki-editor-form").classList.add("hidden");
    document.getElementById("wiki-editor-empty").classList.remove("hidden");
    loadWikiList();
  } catch (e) { toast(e.message, "error"); }
});

document.getElementById("wiki-history-btn").addEventListener("click", async () => {
  if (!currentSlug) return;
  const box = document.getElementById("wiki-history");
  if (!box.classList.contains("hidden")) { box.classList.add("hidden"); return; }
  await renderWikiHistory(box);
});

let wikiDiffBase = null; // 当前选中的对比基线 {id, el}

async function renderWikiHistory(box) {
  try {
    const revs = await api("GET", `/wiki/pages/${currentSlug}/history`);
    wikiDiffBase = null;
    box.classList.remove("hidden");
    let html = "<h4>版本历史</h4>";
    if (revs.length === 0) {
      box.innerHTML = html + "<p>暂无历史</p>";
      return;
    }
    html += '<div id="wiki-diff-panel"></div>';
    html += revs.map((r, i) => `<div class="wiki-rev-item" data-rev="${r.id}">
      <div class="wiki-rev-head">
        <span class="wiki-rev-time">${fmtTime(r.created_at)}</span>
        <span class="wiki-rev-summary">${escapeHtml(r.summary || "（无摘要）")}</span>
        <span class="wiki-rev-actions">
          <button class="mini-btn" data-act="view">查看</button>
          <button class="mini-btn" data-act="diff">对比</button>
          ${i === 0 ? "" : '<button class="mini-btn danger" data-act="revert">回滚到此版本</button>'}
        </span>
      </div>
    </div>`).join("");
    box.innerHTML = html;

    box.querySelectorAll(".wiki-rev-item").forEach(item => {
      const revId = item.dataset.rev;
      item.querySelector('[data-act="view"]').addEventListener("click", () => viewRevision(item, revId));
      item.querySelector('[data-act="diff"]').addEventListener("click", () => pickDiffBase(item, revId));
      const revertBtn = item.querySelector('[data-act="revert"]');
      if (revertBtn) revertBtn.addEventListener("click", () => revertRevision(revId));
    });
  } catch (e) { toast(e.message, "error"); }
}

async function viewRevision(item, revId) {
  let detail = item.querySelector(".wiki-rev-detail");
  if (detail) { detail.remove(); return; }
  try {
    const data = await api("GET", `/wiki/pages/${currentSlug}/revisions/${revId}`);
    detail = document.createElement("div");
    detail.className = "wiki-rev-detail";
    detail.textContent = data.content || "（空页面）";
    item.appendChild(detail);
  } catch (e) { toast(e.message, "error"); }
}

function pickDiffBase(item, revId) {
  const panel = document.getElementById("wiki-diff-panel");
  if (wikiDiffBase && wikiDiffBase.id === revId) {
    wikiDiffBase = null;
    if (panel) panel.innerHTML = "";
    item.querySelector('[data-act="diff"]').classList.remove("danger");
    toast("已取消对比选择", "info");
    return;
  }
  if (!wikiDiffBase) {
    wikiDiffBase = { id: revId, el: item };
    item.querySelector('[data-act="diff"]').classList.add("danger");
    toast("已选择基线版本，请点击另一版本的「对比」", "info");
    return;
  }
  // 已选基线：拉取两版正文并渲染 diff
  const base = wikiDiffBase;
  base.el.querySelector('[data-act="diff"]').classList.remove("danger");
  wikiDiffBase = null;
  showRevisionDiff(base.id, revId, panel);
}

async function showRevisionDiff(baseId, targetId, panel) {
  try {
    const [a, b] = await Promise.all([
      api("GET", `/wiki/pages/${currentSlug}/revisions/${baseId}`),
      api("GET", `/wiki/pages/${currentSlug}/revisions/${targetId}`),
    ]);
    const lines = lineDiff(a.content || "", b.content || "");
    const body = lines.map(l =>
      `<div class="diff-line diff-${l.type}"><span class="diff-gutter">${
        l.type === "ctx" ? " " : l.type === "add" ? "+" : "-"
      }</span>${escapeHtml(l.text) || " "}</div>`).join("");
    panel.innerHTML = `<div class="wiki-diff">
      <div class="wiki-diff-head"><span>版本对比：${baseId.slice(0, 8)} → ${targetId.slice(0, 8)}</span>
      <button class="mini-btn" id="wiki-diff-close">关闭</button></div>
      <div class="wiki-diff-body">${body || '<div class="diff-line diff-ctx">（两版内容相同）</div>'}</div>
    </div>`;
    document.getElementById("wiki-diff-close").addEventListener("click", () => { panel.innerHTML = ""; });
    panel.scrollIntoView({ behavior: "smooth" });
  } catch (e) { toast(e.message, "error"); }
}

// 行级 LCS diff：返回 [{type: "add"|"del"|"ctx", text}]
function lineDiff(oldText, newText) {
  const a = oldText.split("\n");
  const b = newText.split("\n");
  const n = a.length, m = b.length;
  // LCS 为 O(n*m) 内存，超大文档退化为"整块替换"对比
  if (n * m > 4_000_000) {
    return [
      ...a.map(text => ({ type: "del", text })),
      ...b.map(text => ({ type: "add", text })),
    ];
  }
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push({ type: "ctx", text: a[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push({ type: "del", text: a[i] }); i++; }
    else { out.push({ type: "add", text: b[j] }); j++; }
  }
  while (i < n) { out.push({ type: "del", text: a[i++] }); }
  while (j < m) { out.push({ type: "add", text: b[j++] }); }
  return out;
}

async function revertRevision(revId) {
  if (!confirm(`确定回滚到版本 ${revId.slice(0, 8)} 吗？\n将以该版本内容创建一个新版本，历史记录不会丢失。`)) return;
  try {
    await api("POST", `/wiki/pages/${currentSlug}/revert`, { revision_id: revId, summary: "" });
    toast("回滚成功", "success");
    await loadWikiPage(currentSlug);
    const box = document.getElementById("wiki-history");
    await renderWikiHistory(box);
  } catch (e) { toast(e.message, "error"); }
}

// ── Wiki 文档导入 ─────────────────────────────────────────────────
const wikiImportInput = document.getElementById("wiki-import-file");
document.getElementById("wiki-import-btn").addEventListener("click", () => wikiImportInput.click());
wikiImportInput.addEventListener("change", async () => {
  if (!wikiImportInput.files.length) return;
  const file = wikiImportInput.files[0];
  const form = new FormData();
  form.append("file", file);
  try {
    const page = await api("POST", "/wiki/import", form, true);
    toast(`已导入「${page.title}」`, "success");
    wikiImportInput.value = "";
    await loadWikiList();
    loadWikiPage(page.slug);
  } catch (e) { toast(e.message, "error"); }
});

// ── Wiki 附件 ─────────────────────────────────────────────────────
async function loadWikiAttachments(slug) {
  const box = document.getElementById("wiki-attachments");
  const list = document.getElementById("wiki-attachment-list");
  try {
    const atts = await api("GET", `/wiki/pages/${slug}/attachments`);
    list.innerHTML = "";
    if (atts.length === 0) {
      box.classList.add("hidden");
      return;
    }
    box.classList.remove("hidden");
    for (const a of atts) {
      const li = document.createElement("li");
      li.className = "wiki-attachment-item";
      li.innerHTML = `
        <span class="att-name">📎 ${escapeHtml(a.filename)}</span>
        <span class="att-size">${fmtSize(a.size_bytes)}</span>
        <span class="att-actions">
          <button class="att-download btn-secondary" data-id="${a.id}">下载</button>
          <button class="att-delete btn-danger" data-id="${a.id}">删除</button>
        </span>`;
      list.appendChild(li);
    }
  } catch (e) { toast(e.message, "error"); }
}

async function downloadAttachment(id) {
  try {
    const res = await fetch(`${API}/wiki/attachments/${id}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || `下载失败 (${res.status})`);
    }
    const blob = await res.blob();
    const disp = res.headers.get("Content-Disposition") || "";
    const match = disp.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
    const filename = match ? decodeURIComponent(match[1]) : `attachment-${id}`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) { toast(e.message, "error"); }
}

const wikiAttachmentInput = document.getElementById("wiki-attachment-file");
document.getElementById("wiki-attachment-btn").addEventListener("click", () => wikiAttachmentInput.click());
wikiAttachmentInput.addEventListener("change", async () => {
  if (!currentSlug || !wikiAttachmentInput.files.length) return;
  const file = wikiAttachmentInput.files[0];
  const form = new FormData();
  form.append("file", file);
  try {
    await api("POST", `/wiki/pages/${currentSlug}/attachments`, form, true);
    toast("附件上传成功", "success");
    wikiAttachmentInput.value = "";
    loadWikiAttachments(currentSlug);
  } catch (e) { toast(e.message, "error"); }
});

// 附件列表事件委托：下载 / 删除
document.getElementById("wiki-attachment-list").addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const id = btn.dataset.id;
  if (btn.classList.contains("att-download")) {
    downloadAttachment(id);
  } else if (btn.classList.contains("att-delete")) {
    if (!confirm("确定删除该附件吗？")) return;
    try {
      await api("DELETE", `/wiki/attachments/${id}`);
      toast("附件已删除", "success");
      loadWikiAttachments(currentSlug);
    } catch (err) { toast(err.message, "error"); }
  }
});

// ── 知识库 ────────────────────────────────────────────────────────────
const kbFileInput = document.getElementById("kb-file");
const kbDrop = document.getElementById("kb-drop");

kbDrop.addEventListener("dragover", e => { e.preventDefault(); kbDrop.style.borderColor = "#667eea"; });
kbDrop.addEventListener("dragleave", () => { kbDrop.style.borderColor = "#ccc"; });
kbDrop.addEventListener("drop", e => {
  e.preventDefault();
  kbDrop.style.borderColor = "#ccc";
  uploadFiles(e.dataTransfer.files);
});
kbFileInput.addEventListener("change", e => uploadFiles(e.target.files));

async function uploadFiles(files) {
  if (!files.length) return;
  const msg = document.getElementById("kb-upload-msg");
  msg.innerHTML = "";
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      await api("POST", "/knowledge/upload", form, true);
      msg.innerHTML += `<div style="color:#27ae60">✓ ${file.name} 上传成功</div>`;
    } catch (e) {
      msg.innerHTML += `<div style="color:#e74c3c">✗ ${file.name}: ${e.message}</div>`;
    }
  }
  loadKbFiles();
}

async function loadKbFiles() {
  try {
    const files = await api("GET", "/knowledge/files");
    const ul = document.getElementById("kb-file-list");
    ul.innerHTML = "";
    if (files.length === 0) {
      ul.innerHTML = '<li style="text-align:center;color:#999;padding:20px;">暂无文件</li>';
      return;
    }
    files.forEach(f => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="kb-file-actions">
        <span class="kb-file-name">${escapeHtml(f.source_id)}</span>
        <span class="kb-file-size">${fmtSize(f.size_bytes)}</span>
        <button class="kb-file-del" title="删除文件">🗑</button>
      </span>`;
      li.querySelector(".kb-file-del").addEventListener("click", async () => {
        if (!confirm(`确定删除文件「${f.source_id}」吗？\n相关索引将一并刷新。`)) return;
        try {
          await api("DELETE", `/knowledge/files/${f.source_id.split("/").map(encodeURIComponent).join("/")}`);
          toast("文件已删除", "success");
          loadKbFiles();
        } catch (e) { toast(e.message, "error"); }
      });
      ul.appendChild(li);
    });
  } catch (e) { toast(e.message, "error"); }
}

// ── 管理（平台管理员：全局技能上传/列表/删除） ───────────────────────
async function loadAdminSkills() {
  try {
    const skills = await api("GET", "/api/admin/skills");
    const ul = document.getElementById("admin-skill-list");
    ul.innerHTML = "";
    if (skills.length === 0) {
      ul.innerHTML = '<li style="text-align:center;color:#999;padding:20px;">暂无全局技能</li>';
      return;
    }
    skills.forEach(s => {
      const li = document.createElement("li");
      const status = s.valid
        ? `${s.files.length} 个文件 · 更新于 ${fmtTime(s.updated_at)}`
        : `<span style="color:#e74c3c">解析失败：${escapeHtml(s.error)}</span>`;
      li.innerHTML = `<span class="kb-file-actions">
        <span class="kb-file-name" title="${escapeHtml(s.description)}">${escapeHtml(s.name)}</span>
        <span class="kb-file-size">${status}</span>
        <button class="kb-file-del" title="删除技能">🗑</button>
      </span>`;
      li.querySelector(".kb-file-del").addEventListener("click", async () => {
        if (!confirm(`确定删除全局技能「${s.name}」吗？\n删除后下一轮对话起对所有租户不再可用。`)) return;
        try {
          await api("DELETE", `/api/admin/skills/${encodeURIComponent(s.name)}`);
          toast("技能已删除", "success");
          loadAdminSkills();
        } catch (e) { toast(e.message, "error"); }
      });
      ul.appendChild(li);
    });
  } catch (e) { toast(e.message, "error"); }
}

// ── 技能上传：先本地快检 + 服务端预检，通过后才放开上传按钮 ─────────────
const SKILL_MAX_BYTES = 10 * 1024 * 1024; // 与后端 MAX_SKILL_ZIP_BYTES 一致
let adminSkillValidated = false; // 预检通过后才允许点上传

function _adminValidateBox() { return document.getElementById("admin-skill-validate"); }
function _adminUploadBtn() { return document.getElementById("admin-skill-upload-btn"); }

function _resetAdminSkillSelection(noteHtml) {
  adminSkillValidated = false;
  _adminUploadBtn().disabled = true;
  _adminValidateBox().innerHTML = noteHtml || "";
}

// 选择文件后：先本地快检，再调后端预检端点，全部通过才放开上传按钮
document.getElementById("admin-skill-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  document.getElementById("admin-skill-msg").innerHTML = "";
  if (!file) { _resetAdminSkillSelection(); return; }

  // ① 本地快检（不发请求）：后缀、非空、大小上限、zip 魔数 PK\x03\x04
  if (!file.name.toLowerCase().endsWith(".zip")) {
    _resetAdminSkillSelection('<div style="color:#e74c3c">✗ 仅接受 .zip 技能包</div>');
    return;
  }
  if (file.size === 0) {
    _resetAdminSkillSelection('<div style="color:#e74c3c">✗ 文件内容为空</div>');
    return;
  }
  if (file.size > SKILL_MAX_BYTES) {
    _resetAdminSkillSelection('<div style="color:#e74c3c">✗ 压缩包超过 10MB 上限</div>');
    return;
  }
  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  if (!(head[0] === 0x50 && head[1] === 0x4B)) {
    _resetAdminSkillSelection('<div style="color:#e74c3c">✗ 文件不是合法的 zip 压缩包</div>');
    return;
  }

  // ② 服务端预检：与安装完全一致的全量校验，不写盘
  _adminValidateBox().innerHTML = '<div style="color:#888">正在验证技能包…</div>';
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await api("POST", "/api/admin/skills/validate", form, true);
    if (!result.valid) {
      const items = result.errors.map(err => `<li>${escapeHtml(err)}</li>`).join("");
      _resetAdminSkillSelection(`<div style="color:#e74c3c">✗ 验证未通过：<ul style="margin:4px 0 0 18px;">${items}</ul></div>`);
      return;
    }
    const warns = result.warnings.map(w => `<div style="color:#e67e22">⚠ ${escapeHtml(w)}</div>`).join("");
    _adminValidateBox().innerHTML =
      `<div style="color:#27ae60">✓ 验证通过：技能「${escapeHtml(result.name)}」（${result.files.length} 个文件）</div>` +
      `<div style="color:#666;font-size:13px;">${escapeHtml(result.description)}</div>` + warns;
    adminSkillValidated = true;
    _adminUploadBtn().disabled = false;
  } catch (err) {
    _resetAdminSkillSelection(`<div style="color:#e74c3c">✗ 预检请求失败：${escapeHtml(err.message)}</div>`);
  }
});

document.getElementById("admin-skill-upload-btn").addEventListener("click", async () => {
  const fileInput = document.getElementById("admin-skill-file");
  const msg = document.getElementById("admin-skill-msg");
  const file = fileInput.files[0];
  // 双保险：未经过预检的文件一律先验证
  if (!file || !adminSkillValidated) {
    msg.innerHTML = '<div style="color:#e74c3c">请先选择 zip 技能包并等待验证通过</div>';
    return;
  }
  const form = new FormData();
  form.append("file", file);
  form.append("overwrite", document.getElementById("admin-skill-overwrite").checked);
  try {
    const info = await api("POST", "/api/admin/skills/upload", form, true);
    msg.innerHTML = `<div style="color:#27ae60">✓ 技能「${escapeHtml(info.name)}」已安装，下一轮对话自动加载到模型</div>`;
    fileInput.value = "";
    _resetAdminSkillSelection();
    loadAdminSkills();
  } catch (e) {
    msg.innerHTML = `<div style="color:#e74c3c">✗ ${escapeHtml(e.message)}</div>`;
  }
});

// ── 对话（会话侧边栏 + SSE 流式 + 来源引用） ──────────────────────────
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
let currentThreadId = null;
let isStreaming = false;

chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", e => { if (e.key === "Enter") sendChat(); });
document.getElementById("thread-new-btn").addEventListener("click", startNewThread);

async function loadThreadList() {
  try {
    const threads = await api("GET", "/threads");
    const ul = document.getElementById("thread-list");
    ul.innerHTML = "";
    if (threads.length === 0) {
      ul.innerHTML = '<li class="thread-empty">暂无会话</li>';
      return;
    }
    threads.forEach(t => {
      const li = document.createElement("li");
      li.className = "thread-item" + (t.id === currentThreadId ? " active" : "");
      li.innerHTML = `<span class="thread-title" title="${escapeHtml(t.title)}">${escapeHtml(t.title)}</span>
        <button class="thread-del" title="删除会话">✕</button>`;
      li.querySelector(".thread-title").addEventListener("click", () => selectThread(t.id));
      li.querySelector(".thread-del").addEventListener("click", async e => {
        e.stopPropagation();
        if (!confirm(`确定删除会话「${t.title}」吗？`)) return;
        try {
          await api("DELETE", `/threads/${t.id}`);
          if (currentThreadId === t.id) startNewThread();
          loadThreadList();
        } catch (err) { toast(err.message, "error"); }
      });
      ul.appendChild(li);
    });
  } catch (e) { toast(e.message, "error"); }
}

function startNewThread() {
  currentThreadId = null;
  document.querySelectorAll(".thread-item").forEach(li => li.classList.remove("active"));
  chatMessages.innerHTML = '<div class="chat-welcome"><h3>👋 开始一个新对话</h3>'
    + "<p>在下方输入问题，将自动创建会话。</p></div>";
  // 新会话尚未创建，但首条消息发送时会自动建会话（见 sendChat），
  // 因此输入框必须立即可用，否则用户无法输入第一条消息
  if (!isStreaming) {
    chatInput.disabled = false;
    chatSend.disabled = false;
  }
}

async function selectThread(id) {
  currentThreadId = id;
  // 高亮交给 loadThreadList 重渲染
  try {
    const detail = await api("GET", `/threads/${id}`);
    chatMessages.innerHTML = "";
    (detail.messages || []).forEach(m => appendMsg(m.role, m.text || m.content || ""));
    chatInput.disabled = false;
    chatSend.disabled = false;
    loadThreadList();
  } catch (e) { toast(e.message, "error"); }
}

async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg || isStreaming) return;

  // 首条消息时创建会话（默认标题，后端首轮自动用问题重命名）
  if (!currentThreadId) {
    try {
      const t = await api("POST", "/threads", { title: "新对话" });
      currentThreadId = t.id;
      chatInput.disabled = false;
      chatSend.disabled = false;
    } catch (e) { toast("创建会话失败: " + e.message, "error"); return; }
  }

  appendMsg("user", msg);
  chatInput.value = "";
  const assistantEl = createAssistantBubble();
  setStreaming(true);

  try {
    await streamChat(msg, assistantEl);
  } finally {
    setStreaming(false);
    loadThreadList();
  }
}

function setStreaming(on) {
  isStreaming = on;
  chatInput.disabled = !currentThreadId || on;
  chatSend.disabled = !currentThreadId || on;
}

function createAssistantBubble() {
  const el = document.createElement("div");
  el.className = "chat-msg assistant streaming";
  const text = document.createElement("div");
  text.className = "chat-text";
  const sources = document.createElement("div");
  sources.className = "chat-sources hidden";
  el.appendChild(text);
  el.appendChild(sources);
  chatMessages.appendChild(el);
  return { el, text, sources, answer: "", sourceMap: new Map() };
}

function appendSources(bubble, newSources) {
  for (const s of newSources) {
    if (!s.source) continue;
    bubble.sourceMap.set(s.source, s);
  }
  if (bubble.sourceMap.size === 0) return;
  bubble.sources.classList.remove("hidden");
  const chips = [...bubble.sourceMap.values()].map(s => {
    const score = typeof s.score === "number" ? s.score.toFixed(3) : "";
    const section = s.section ? ` · ${escapeHtml(s.section)}` : "";
    const tier = s.tier === "wiki"
      ? '<span class="source-tier t-wiki">wiki</span>'
      : (s.tier === "raw" ? '<span class="source-tier t-raw">原始</span>' : "");
    return `<span class="source-chip">📄 ${escapeHtml(s.source)}${section}`
      + tier
      + (score ? `<span class="source-score">${score}</span>` : "") + "</span>";
  }).join("");
  bubble.sources.innerHTML = `<div class="chat-sources-title">参考来源（${bubble.sourceMap.size}）</div>${chips}`;
}

async function streamChat(message, bubble) {
  const res = await fetch("/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ thread_id: currentThreadId, message }),
  });
  if (!res.ok || !res.body) {
    let detail = `流式请求失败 (${res.status})`;
    try { detail = (await res.json()).detail || detail; } catch {}
    bubble.answer = "出错: " + detail;
    bubble.text.textContent = bubble.answer;
    bubble.el.classList.remove("streaming");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE 事件以空行分隔
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      handleSseEvent(rawEvent, bubble);
    }
  }
  bubble.el.classList.remove("streaming");
}

function handleSseEvent(raw, bubble) {
  let event = "message";
  const dataLines = [];
  raw.split("\n").forEach(line => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  });
  if (dataLines.length === 0) return;
  let data;
  try { data = JSON.parse(dataLines.join("\n")); } catch { return; }

  if (event === "token" && data.text) {
    bubble.answer += data.text;
    bubble.text.textContent = bubble.answer;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  } else if (event === "sources") {
    appendSources(bubble, data.sources || []);
  } else if (event === "interrupt") {
    // HITL：代理在敏感工具前挂起，渲染审批卡片
    renderApprovalCard(data.interrupts || []);
  } else if (event === "error") {
    bubble.answer += (bubble.answer ? "\n\n" : "") + "⚠ " + (data.detail || "执行失败");
    bubble.text.textContent = bubble.answer;
  }
  // done：无额外动作
}

// ── HITL 审批卡片（简洁行式） ─────────────────────────────────────────
const TOOL_LABELS = {
  forget_memory: "删除长期记忆",
  write_file: "写入文件（长期记忆）",
  edit_file: "修改文件（长期记忆）",
};

// 从参数里提炼一行摘要：优先路径/fact_id，其次 content 截断，兜底紧凑 JSON
function summarizeArgs(args) {
  if (!args || typeof args !== "object") return "";
  if (typeof args.path === "string") return args.path;
  if (typeof args.fact_id === "string") return `fact_id: ${args.fact_id}`;
  if (typeof args.content === "string") {
    return args.content.length > 60 ? args.content.slice(0, 60) + "…" : args.content;
  }
  const s = JSON.stringify(args);
  return s.length > 80 ? s.slice(0, 80) + "…" : s;
}

function renderApprovalCard(interrupts) {
  if (!interrupts.length) return;
  const welcome = document.getElementById("chat-welcome");
  if (welcome) welcome.remove();
  const card = document.createElement("div");
  card.className = "approval-card";
  const items = interrupts.map(it => {
    const label = TOOL_LABELS[it.tool] || it.tool;
    const summary = escapeHtml(summarizeArgs(it.args));
    const full = escapeHtml(JSON.stringify(it.args, null, 2));
    return `<div class="approval-item">`
      + `<div class="approval-line">`
      + `<span class="approval-tool">${escapeHtml(label)}</span>`
      + (summary ? `<span class="approval-summary" title="${summary}">${summary}</span>` : "")
      + `</div>`
      + `<details class="approval-details"><summary>完整参数</summary>`
      + `<pre class="approval-args">${full}</pre></details></div>`;
  }).join("");
  card.innerHTML = `<div class="approval-head">`
    + `<span>⚠ 待审批 <span class="approval-count">${interrupts.length}</span> 项</span>`
    + `<div class="approval-actions">`
    + `<button class="approval-reject">拒绝</button>`
    + `<button class="approval-approve">批准</button></div></div>`
    + items;
  chatMessages.appendChild(card);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  card.querySelector(".approval-approve").addEventListener("click", () => resumeApproval(card, interrupts, "approve"));
  card.querySelector(".approval-reject").addEventListener("click", () => resumeApproval(card, interrupts, "reject"));
}

async function resumeApproval(card, interrupts, type) {
  card.querySelectorAll("button").forEach(b => b.disabled = true);
  const decisions = interrupts.map(() => type === "approve"
    ? { type: "approve" }
    : { type: "reject", message: "用户拒绝了该操作" });
  try {
    const data = await api("POST", "/chat/resume", { thread_id: currentThreadId, decisions });
    showApprovalResult(card, type === "approve" ? "✓ 已批准，继续执行" : "✕ 已拒绝该操作");
    if (data.status === "pending_approval") {
      renderApprovalCard(data.interrupts || []);
    } else if (data.answer) {
      appendMsg("assistant", data.answer);
    }
  } catch (e) {
    showApprovalResult(card, "⚠ " + e.message);
  }
}

// 决策后整张卡片坍缩为一行结果
function showApprovalResult(card, text) {
  card.classList.add("resolved");
  card.innerHTML = `<div class="approval-result">${escapeHtml(text)}</div>`;
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendMsg(role, text) {
  const welcome = document.getElementById("chat-welcome");
  if (welcome) welcome.remove();
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── 启动 ──────────────────────────────────────────────────────────────
if (token) {
  showApp();
} else {
  document.getElementById("auth-layer").classList.remove("hidden");
}
