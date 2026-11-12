"""桌面端 API 错误类型。"""

from __future__ import annotations


class ApiError(Exception):
    """后端返回的非 2xx 错误，detail 为服务端可读描述。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ApiAuthExpiredError(ApiError):
    """refresh token 失效（刷新轮换失败），需要重新登录。"""

    def __init__(self, detail: str = "登录已过期，请重新登录") -> None:
        super().__init__(401, detail)


class ApiNetworkError(ApiError):
    """网络层错误（连接失败/超时），status_code 固定为 0。"""

    def __init__(self, detail: str) -> None:
        super().__init__(0, detail)
