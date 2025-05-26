import time
from cs336_basics.train_bpe import SPECIAL_TOKEN
from cs336_basics.tokenizer import Tokenizer


def sample_docs():
    ts_tokenizer = Tokenizer.from_pickles(
        "./bpe_tinystories.pkl",
        [SPECIAL_TOKEN],
    )
    owt_tokenizer = Tokenizer.from_pickles(
        "./bpe_expts_owt.pkl",
        [SPECIAL_TOKEN],
    )
    ts_sample = ""
    with open("./data/TinyStoriesV2-GPT4-train.txt", "r") as f:
        docs_count = 0
        while docs_count < 10:
            line = f.readline()
            ts_sample += line
            if SPECIAL_TOKEN in line:
                docs_count += 1
    text_size = len(ts_sample.encode("utf-8"))
    ts_encoded = ts_tokenizer.encode(ts_sample)
    owt_encoded = owt_tokenizer.encode(ts_sample)
    ts_ratio = text_size / len(ts_encoded)
    owt_ratio = text_size / len(owt_encoded)
    assert ts_tokenizer.decode(ts_encoded) == ts_sample
    assert owt_tokenizer.decode(owt_encoded) == ts_sample
    print(f"TinyStories sample size: {text_size} bytes")
    print(f"TinyStories encoded compression ratio: {ts_ratio:0.2f} bytes/token")
    print(f"OWT encoded compression ratio: {owt_ratio:0.2f} bytes/token")
    owt_sample = ""
    with open("./data/owt_train.txt", "r") as f:
        docs_count = 0
        while docs_count < 10:
            line = f.readline()
            owt_sample += line
            if SPECIAL_TOKEN in line:
                docs_count += 1
    text_size = len(owt_sample.encode("utf-8"))
    ts_encoded = ts_tokenizer.encode(owt_sample)
    owt_encoded = owt_tokenizer.encode(owt_sample)
    ts_ratio = text_size / len(ts_encoded)
    owt_ratio = text_size / len(owt_encoded)
    assert ts_tokenizer.decode(ts_encoded) == owt_sample
    assert owt_tokenizer.decode(owt_encoded) == owt_sample
    print(f"OWT sample size: {text_size} bytes")
    print(f"TinyStories encoded compression ratio: {ts_ratio:0.2f} bytes/token")
    print(f"OWT encoded compression ratio: {owt_ratio:0.2f} bytes/token")


def throughput():
    owt_tokenizer = Tokenizer.from_pickles(
        "./bpe_expts_owt.pkl",
        [SPECIAL_TOKEN],
    )
    owt_sample = ""
    with open("./data/owt_train.txt", "r") as f:
        avg_throughput, num_iterations = 0, 10
        for _ in range(num_iterations):
            docs_count = 0
            while docs_count < 1000:
                line = f.readline()
                owt_sample += line
                if SPECIAL_TOKEN in line:
                    docs_count += 1
            text_size = len(owt_sample.encode("utf-8"))
            start = time.time()
            owt_encoded = owt_tokenizer.encode(owt_sample)
            end = time.time() - start
            throughput = text_size / end / 1e6  # MB/s
            avg_throughput += throughput
            print(f"Encoding of {text_size / 1e6} MB took {end:.2f} seconds")
            print(f"Estimated throughput: {throughput:.2f} MB/s")
            assert owt_tokenizer.decode(owt_encoded) == owt_sample
        avg_throughput /= num_iterations
        print(f"Estimated avg throughput: {avg_throughput:.2f} MB/s")


def encode(data_path: str, input_file: str, output_file: str):
    tokenizer = Tokenizer.from_pickles(
        data_path,
        [SPECIAL_TOKEN],
    )
    print(f"Encoding {input_file} with {data_path} tokenizer")
    start = time.time()
    with open(input_file, "r") as i:
        with open(output_file, "w") as o:
            for token in tokenizer.encode_iterable(i.read()):
                o.write(str(token) + "\n")
    print(f"Encoding {input_file} took {time.time() - start:.2f} seconds")


def encode_tinystories():
    encode(
        "./bpe_tinystories.pkl",
        "../data/TinyStoriesV2-GPT4-valid.txt",
        "./tinystories_valid_encoded.txt",
    )
    encode(
        "./bpe_tinystories.pkl",
        "../data/TinyStoriesV2-GPT4-train.txt",
        "./tinystories_train_encoded.txt",
    )


def encode_owt():
    encode(
        "./bpe_expts_owt.pkl",
        "../data/owt_valid.txt",
        "./owt_valid_encoded.txt",
    )
    encode(
        "./bpe_expts_owt.pkl",
        "../data/owt_train.txt",
        "./owt_train_encoded.txt",
    )


if __name__ == "__main__":
    # sample_docs()
    # throughput()
    # encode_tinystories()
    encode_owt()
