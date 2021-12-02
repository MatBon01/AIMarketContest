"""
This type stub file was generated by pyright.
"""

def touch(path): ...

class VideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame. It
    comes with an `enabled` option so you can still use the same code
    on episodes where you don't want to record video.

    Note:
        You are responsible for calling `close` on a created
        VideoRecorder, or else you may leak an encoder process.

    Args:
        env (Env): Environment to take video of.
        path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
        base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
        metadata (Optional[dict]): Contents to save to the metadata file.
        enabled (bool): Whether to actually record video, or just no-op (for convenience)
    """

    def __init__(
        self, env, path=..., metadata=..., enabled=..., base_path=...
    ) -> None: ...
    @property
    def functional(self): ...
    def capture_frame(self):  # -> None:
        """Render the given `env` and add the resulting frame to the video."""
        ...
    def close(self):  # -> None:
        """Flush all data to disk and close any open frame encoders."""
        ...
    def write_metadata(self): ...
    def __del__(self): ...

class TextEncoder:
    """Store a moving picture made out of ANSI frames. Format adapted from
    https://github.com/asciinema/asciinema/blob/master/doc/asciicast-v1.md"""

    def __init__(self, output_path, frames_per_sec) -> None: ...
    def capture_frame(self, frame): ...
    def close(self): ...
    @property
    def version_info(self): ...

class ImageEncoder:
    def __init__(
        self, output_path, frame_shape, frames_per_sec, output_frames_per_sec
    ) -> None: ...
    @property
    def version_info(self): ...
    def start(self): ...
    def capture_frame(self, frame): ...
    def close(self): ...