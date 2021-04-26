from tqdm import tqdm
from skopt.space import Real, Categorical, Integer

class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def create_dimensions_from_parameters (parameters):
    dim1 = Integer(name='k_retriever', low=min(parameters['k_retriever']), high=max(parameters['k_retriever']) + 1)
    dim2 = Integer(name='k_title_retriever', low=min(parameters['k_title_retriever']),
                   high=max(parameters['k_title_retriever']) + 1)
    dim3 = Integer(name='k_reader_per_candidate', low=min(parameters['k_reader_per_candidate']),
                   high=max(parameters['k_reader_per_candidate']) + 1)
    dim4 = Integer(name='k_reader_total', low=min(parameters['k_reader_total']), high=max(parameters['k_reader_total']) + 1)
    dim5 = Categorical(name='reader_model_version', categories=parameters['reader_model_version'])
    dim6 = Categorical(name='retriever_model_version', categories=parameters['retriever_model_version'])
    dim7 = Categorical(name='retriever_type', categories=parameters['retriever_type'])
    dim8 = Categorical(name='squad_dataset', categories=parameters['squad_dataset'])
    dim9 = Categorical(name='filter_level', categories=parameters['filter_level'])
    dim10 = Categorical(name='preprocessing', categories=parameters['preprocessing'])
    dim11 = Integer(name='boosting', low=min(parameters['boosting']), high=max(parameters['boosting']) + 1)
    dim12 = Categorical(name='split_by', categories=parameters['split_by'])
    dim13 = Integer(name='split_length', low=min(parameters['split_length']), high=max(parameters['split_length']) + 1)

    dimensions = [dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9, dim10, dim11, dim12, dim13]

    return dimensions