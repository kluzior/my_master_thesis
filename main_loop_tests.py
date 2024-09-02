import socket
from packages2.start_communication import start_communication
from packages2.loop_state_machine import LoopStateMachine

c, s = start_communication()


loop = LoopStateMachine(c)


loop.run()