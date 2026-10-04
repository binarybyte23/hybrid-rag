"""Run async tests without pytest-asyncio (stdlib only)."""
import asyncio
import inspect


def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker("asyncio")


def pytest_pyfunc_call(pyfuncitem):
    fn = pyfuncitem.function
    if inspect.iscoroutinefunction(fn):
        kwargs = {k: v for k, v in pyfuncitem.funcargs.items()
                  if k in fn.__code__.co_varnames}
        asyncio.run(fn(**kwargs))
        return True
    return None
