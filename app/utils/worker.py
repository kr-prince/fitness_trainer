import multiprocessing as mp
from abc import ABC, abstractmethod


class Worker(ABC):
  """ 
    A generic worker class to implement other class based functionalities using 
    multi-processing
  """

  @abstractmethod
  def main(self):
    # This function has to be implemented by the base class
    pass

  @property
  @abstractmethod
  def name(self):
    # This property will return the name of the base class
    # return "BaseClassName"
    pass


  def run(self):
    # This function starts the main method for this worker after setting up the queues 
    # for communicaton
    self.inputs = mp.Queue()
    self.outputs = mp.Queue()
    self.proc = mp.Process(target=self.main)
    self.proc.start()


  def stop(self):
    # This function will kill the process
    self.inputs.put(0)
    # print("%s:  Inputs(%s)  Outputs(%s)" %(self.name(), self.inputs.empty(), self.outputs.empty()))
    self.proc.join()
    print("%s killed with code %s" %(self.name(), str(self.proc.exitcode)))


  def get_output(self):
    # This function reads and returns from the output queue
    return None if self.outputs.empty() else self.outputs.get()


