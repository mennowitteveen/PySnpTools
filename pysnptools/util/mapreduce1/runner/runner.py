class Runner:
    '''
    !!!!cmk
    A runner is executes a 'distributable' job. FastLmmSet is an example of a distributable job. #!!!cmk update
    Local is an example of a runner. Local can execute FastLMMSet (or any other distributable job) on a local machine as a single process.
    LocalMultiProc is another runner. It can execute FastLmmSet, etc on a multiple processes on a local machine.

    All runners implement the Runner interface (but Python doesn't check or enforce this)

    interface IRunner
        def run(self, distributable):
            # returns the value computed by the distributable job


    '''
    pass
