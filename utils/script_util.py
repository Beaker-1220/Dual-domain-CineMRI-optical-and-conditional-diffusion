import argparse
import pickle


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def load_args_dict(file_name):
    """
    Load a pickle file `file_name` which contains a dict of args

    :param file_name: A pickle file name.
    :return: A dict.
    """
    with open(file_name, "rb") as f:
        return pickle.load(f)


def save_args_dict(args_dict, file_name):
    """
    Save `args_dict` as a pickle file `file_name`

    :param args_dict: A dict contains args.
    :param file_name: A pickle file name.
    :return: None
    """
    with open(file_name, "wb") as f:
        pickle.dump(args_dict, f)
