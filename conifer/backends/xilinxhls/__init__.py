import sys
import importlib.util

SPEC_WRITER = importlib.util.find_spec('.writer', __name__)

writer = importlib.util.module_from_spec(SPEC_WRITER)
if '_tool' in locals() != None:
    writer._tool = _tool
SPEC_WRITER.loader.exec_module(writer)

write = writer.write
auto_config = writer.auto_config
decision_function = writer.decision_function
sim_compile = writer.sim_compile
build = writer.build
_init_model = writer._init_model

del SPEC_WRITER
