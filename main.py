"""Native vision guard - keep one dead image from killing the whole request.

Context
-------
AstrBot's OpenAI-compatible provider rewrites every ``image_url`` block of a
request into a data URL. When a reference cannot be resolved (typically a local
cache path that no longer exists), ``_transform_content_part`` falls back to
returning the ORIGINAL block, so the raw reference is sent to the provider and
the WHOLE request is rejected. DeepSeek answers::

    400 ... Unsupported image_url format

and every later turn of that conversation keeps failing.

This plugin replaces that single method at runtime, so an unresolvable image
degrades to a ``[Image unavailable]`` text block instead. Everything else -
resolvable images, audio blocks, text blocks - is delegated to the original
implementation untouched.

Why a plugin instead of editing the source file: AstrBot upgrades replace the
program directory but never ``data/plugins``, so this survives upgrades with no
re-patching.

Guarantees
----------
* Delegation: non-image blocks and successfully resolved images go through the
  original method.
* No clobbering: if the installed AstrBot no longer contains the buggy fallback
  (upstream fixed it), the patch is NOT applied.
* Loud failure: if the target class or method cannot be located, a warning is
  logged and nothing is changed - never a silent no-op.
* Idempotent: re-initializing does not stack patches.

Status: send ``/vision_guard`` in any chat the bot listens to.
"""

import inspect

try:
    from astrbot.api import logger
except ImportError:  # pragma: no cover - only for out-of-tree imports
    import logging

    logger = logging.getLogger(__name__)

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_native_vision_guard"
PATCH_FLAG = "_native_vision_guard_patched"
BUGGY_MARKER = "return resolved_part or part"
TARGET_MODULE = "astrbot.core.provider.sources.openai_source"
TARGET_METHOD = "_transform_content_part"
PLACEHOLDER = {"type": "text", "text": "[Image unavailable]"}

STATE = {"active": False, "detail": "not initialized"}


def _find_target_class():
    """Locate the provider class that owns the target method.

    Returns:
        A ``(class, name, error)`` tuple; ``error`` is None on success.
    """
    try:
        module = __import__(TARGET_MODULE, fromlist=["*"])
    except Exception as exc:
        return None, None, "cannot import %s: %r" % (TARGET_MODULE, exc)

    classes = [
        (name, obj)
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and getattr(obj, "__module__", None) == module.__name__
    ]
    owning = [(n, o) for n, o in classes if TARGET_METHOD in vars(o)]
    pool = owning or [(n, o) for n, o in classes if hasattr(o, TARGET_METHOD)]
    if not pool:
        return None, None, "no class in %s defines %s" % (TARGET_MODULE, TARGET_METHOD)
    return pool[0][1], pool[0][0], None


def _build_patched(original):
    """Wrap the original method so unresolvable images never reach the wire."""

    async def _transform_content_part(self, part):
        if not isinstance(part, dict) or part.get("type") != "image_url":
            return await original(self, part)

        url, image_detail = self._extract_image_part_info(part)
        if not url:
            return part

        try:
            resolved_part = await self._resolve_image_part(
                url, image_detail=image_detail
            )
        except Exception as exc:
            logger.warning(
                "native_vision_guard: image %s could not be prepared; "
                "dropping this image block. error: %r",
                url,
                exc,
            )
            return dict(PLACEHOLDER)

        if resolved_part is not None:
            return resolved_part

        logger.warning(
            "native_vision_guard: image preprocessing returned nothing; "
            "dropping this image block (%s).",
            url,
        )
        return dict(PLACEHOLDER)

    setattr(_transform_content_part, PATCH_FLAG, True)
    return _transform_content_part


def apply_patch():
    """Apply the guard once.

    Returns:
        A ``(ok, detail)`` tuple describing the outcome.
    """
    cls, cls_name, error = _find_target_class()
    if error:
        return False, error

    original = getattr(cls, TARGET_METHOD)
    if getattr(original, PATCH_FLAG, False):
        return True, "already patched (%s.%s)" % (cls_name, TARGET_METHOD)

    try:
        source = inspect.getsource(original)
    except Exception as exc:
        source = ""
        logger.warning(
            "native_vision_guard: cannot read source of %s.%s (%r); applying "
            "the guard without the upstream-fix check.",
            cls_name,
            TARGET_METHOD,
            exc,
        )

    if source and BUGGY_MARKER not in source:
        return False, (
            "upstream no longer contains the buggy fallback in %s.%s; "
            "not overriding it" % (cls_name, TARGET_METHOD)
        )

    setattr(cls, TARGET_METHOD, _build_patched(original))
    return True, "patched %s.%s" % (cls_name, TARGET_METHOD)


@register(
    PLUGIN_NAME,
    "local",
    "Keep one unresolvable image reference from failing the whole LLM request",
    "1.0.0",
)
class NativeVisionGuard(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        ok, detail = apply_patch()
        STATE["active"] = ok
        STATE["detail"] = detail
        if ok:
            logger.info("[native_vision_guard] ACTIVE - %s", detail)
        else:
            logger.warning("[native_vision_guard] NOT ACTIVE - %s", detail)

    async def terminate(self):
        STATE["active"] = False
        STATE["detail"] = "plugin terminated"

    @filter.command("vision_guard")
    async def vision_guard_status(self, event: AstrMessageEvent):
        yield event.plain_result(
            "native_vision_guard: %s - %s"
            % ("ACTIVE" if STATE["active"] else "INACTIVE", STATE["detail"])
        )
