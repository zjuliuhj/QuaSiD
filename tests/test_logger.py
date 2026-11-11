import sys
from pathlib import Path
from types import SimpleNamespace

from tensorboard.compat.proto.event_pb2 import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.logger import Logger


def test_tensorboard_writes_locally_and_archives_after_close(tmp_path, monkeypatch):
    live_root = tmp_path / 'local-disk'
    save_dir = tmp_path / 'shared-storage' / 'run' / 'log'
    monkeypatch.setenv('QUASID_TENSORBOARD_ROOT', str(live_root))
    logger = Logger(SimpleNamespace(save_dir=str(save_dir), tensorboard=True))
    logger.add_scalar(tag='train/loss', scalar_value=0.25, global_step=50)
    logger.flush()

    live_dir = logger.tb_dir
    assert live_dir.is_relative_to(live_root)
    assert list(live_dir.glob('events.out.tfevents.*'))
    assert not list((save_dir / 'logs').glob('events.out.tfevents.*'))

    logger.close()

    event_file, = live_dir.glob('events.out.tfevents.*')
    archived = save_dir / 'logs' / event_file.name
    assert archived.read_bytes() == event_file.read_bytes()
    scalars = []
    with archived.open('rb') as stream:
        while length := stream.read(8):
            stream.read(4)  # TFRecord length checksum.
            event = Event.FromString(stream.read(int.from_bytes(length, 'little')))
            stream.read(4)  # TFRecord payload checksum.
            scalars.extend(
                (event.step, value.simple_value)
                for value in event.summary.value if value.tag == 'train/loss'
            )
    assert scalars == [(50, 0.25)]
