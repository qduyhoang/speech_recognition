import enchant

right = False
d = enchant.Dict("en_US")
d.check("correctcorrectso")
print(d.check("correctcorrectso"))