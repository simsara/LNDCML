from utils import env


def prepare_luna():
    luna_segment = env.get('luna_segment')
    savepath = env.get('preprocess_result_path')
    luna_data = env.get('luna_data')
    luna_label = env.get('luna_label')


if __name__ == '__main__':
    prepare_luna()
