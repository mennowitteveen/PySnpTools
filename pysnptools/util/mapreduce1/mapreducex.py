from pysnptools.util.mapreduce1.mapreduce import _identity, map_reduce, _dyn_vars
from pysnptools.util.mapreduce1.runner import Local

#!!!improve the name?
def map_reduceX(input_seq, mapper=_identity, reducer=list, runner=None,name=None):
    '''
    !!!cmk update
    Function for running a function on sequence of inputs and running a second function on the results. Can be nested and clusterized.
    For each top-level input, a separate job will be created.


    :param input_seq: a sequence of inputs. The sequence must support the len function and be indexable. e.g. a list, xrange(100)
    :type input_seq: a sequence

    :param mapper: A function to apply to each set of inputs (optional). Defaults to the identity function. (Also see 'mapper')
    :type mapper: a function

    :param reducer: A function to turn the results from the mapper to a single value (optional). Defaults to creating a list of the results.
    :type reducer: a function that takes a sequence

    :param name: A name to be displayed if this work is done on a cluster.
    :type name: a string

    :param runner: a runner, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: a runner.

    :rtype: The results from the reducer.

    '''
    if runner is None:
        runner = Local()

    if name==None:
        name = str(distributable_list[0]) or ""
        if len(distributable_list) > 1:
            name += "-etc"

    with _dyn_vars(is_in_nested=True):
        distributable_list = [mapper(input) for input in input_seq]
    if hasattr(runner,"run_list"):
        return runner.run_list(distributable_list,reducer=reducer,name=name)
    else:
        result_list = [runner.run(distributable) for distributable in distributable_list]
        top_level_result = reducer(result_list)
        return top_level_result

