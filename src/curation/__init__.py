"""LLM Wiki 自动沉淀层。

文档摄入后由 LLM 通读，自动产出/更新结构化 Wiki 页面：

- ``markdown.py``：页面模板、slug 生成、frontmatter 构造、矛盾区块、
  index/log 渲染（纯函数，不碰 IO）；
- ``llm.py``：结构化提取器（摘要+实体+矛盾信号），模型可注入；
- ``service.py``：沉淀编排（读全文→提取→匹配合并→upsert 页面→
  矛盾标记→index/log→刷新索引）；
- ``scheduler.py``：上传后 fire-and-forget 调度（开关/去重/超时）。

事实源仍是 Wiki 业务库，镜像与索引为派生数据；本包只通过
``src.persistence.database`` 与 ``src.rag.knowledge_contract`` 访问数据，
不直接改原始文档。
"""
