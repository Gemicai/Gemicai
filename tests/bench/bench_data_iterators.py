# Here we can put benchmark tests
import gemicai as gem

# This might be an interesting library for benchmarking, or maybe it's a bit over the top for now, haven't decided yet.
# import pycallgraph

# Otherwise just use datetime
from datetime import datetime

label = 'BodyPartExamined'
bench_dataset = gem.DicomoDataset.get_dicomo_dataset('examples', labels=[label])


def bench_iter(dataset):
    assert isinstance(dataset, gem.GemicaiDataset)
    start = datetime.now()
    for _ in dataset:
        pass
    print('bench_iter took {}'.format(datetime.now() - start))


def bench_classes(dataset):
    assert isinstance(dataset, gem.GemicaiDataset)
    start = datetime.now()
    _ = dataset.classes(label)
    print('bench_classes took {}'.format(datetime.now() - start))


if __name__ == '__main__':
    bench_iter(bench_dataset)
    bench_classes(bench_dataset)
