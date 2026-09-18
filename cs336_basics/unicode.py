import sys

try:
	sys.stdout.reconfigure(encoding='utf-8')
except Exception:
	pass

# chr(0)
# print(chr(0))

# print(repr(chr(0)))
# "this is a test" + chr(0) + "string"
# print("this is a test" + chr(0) + "string")
# print("this is a test" + repr(chr(0)) + "string")

print("————————————————————Unicode编码——————————————————————————————")
print(bytes([1]))
