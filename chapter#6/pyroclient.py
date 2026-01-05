import Pyro4

name = input("hiya blud! your good name? ").strip()

server = Pyro4.Proxy("PYRONAME:server")
print(server.welcomeMessage(name))