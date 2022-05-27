import SCons


def DistributedSetup(instance_type=None, ami_id=None):
	"""
	"""
	pass


def DistributedCommand(target, source, env):
	"""
	"""
	pass


env.Append(BUILDERS={"DistributedCommand": Builder(action=DistributedCommand)})
