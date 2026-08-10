from unittest.mock import patch

from gguf.utility import RemoteTensor, SafetensorRemote


def test_remote_safetensors_uses_requested_revision() -> None:
    revision = "refs/pr/123"
    expected_prefix = "https://huggingface.co/org/model/resolve/refs%2Fpr%2F123/"
    index = b'{"weight_map":{"tensor":"model-00001-of-00001.safetensors"}}'
    calls: list[tuple[str, int, int]] = []

    def check_file_exist(url: str) -> bool:
        return url.endswith("model.safetensors.index.json")

    def get_data_by_range(url: str, start: int, size: int = -1) -> bytes:
        calls.append((url, start, size))
        return index

    with (
        patch.object(SafetensorRemote, "check_file_exist", side_effect=check_file_exist),
        patch.object(SafetensorRemote, "get_data_by_range", side_effect=get_data_by_range),
        patch.object(SafetensorRemote, "get_list_tensors", return_value={}) as list_tensors,
    ):
        SafetensorRemote.get_list_tensors_hf_model("org/model", revision=revision)

    assert calls == [(expected_prefix + "model.safetensors.index.json", 0, -1)]
    list_tensors.assert_called_once_with(expected_prefix + "model-00001-of-00001.safetensors")


class _Response:
    def __init__(self, data: bytes):
        self.data = data
        self.status_code = 200
        self.headers = {"Content-Length": str(len(data))}

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.data

    def close(self) -> None:
        pass

    @property
    def content(self) -> bytes:
        return self.data


def test_remote_tensor_full_file_cache(tmp_path) -> None:
    response = _Response(b"abcdefgh")
    tensor = RemoteTensor(dtype="U8", shape=(3,), offset_start=2, size=3, url="https://example.test/shard")

    SafetensorRemote.configure_file_cache(tmp_path)
    try:
        with patch("requests.get", return_value=response) as get:
            with SafetensorRemote.full_file_cache():
                assert tensor.data() == b"cde"
                assert tensor.data() == b"cde"
        get.assert_called_once()
    finally:
        SafetensorRemote.configure_file_cache(None)


def test_remote_range_is_inclusive_without_extra_byte() -> None:
    response = _Response(b"abc")
    with patch("requests.get", return_value=response) as get:
        assert SafetensorRemote.get_data_by_range("https://example.test/shard", 10, 3) == b"abc"

    assert get.call_args.kwargs["headers"]["Range"] == "bytes=10-12"
