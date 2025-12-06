while True:
	user = input("user: ")
	ai = input("ai: ")
	with open("data/pairs.txt", "a") as file:
		file.write("user: " + user + "\n")
		file.write("ai:" + ai + "\n")
