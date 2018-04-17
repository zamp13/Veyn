from __future__ import print_function
import random, json, sys

class Param:
    def __init__(self, value):
        self.set(value)

    def generate(self):
        pass

    def get(self):
        return self.value

    def set(self, value):
        if hasattr(self, 'value') and type(value) != type(self.value):
            value = type(self.value)(value)
        self.value = value

    def __str__(self):
        return str(self.value)


class Choice(Param):
    def __init__(self, values, default=None):
        self.values = values
        self.set(default)

    def generate(self):
        self.set(random.choice(self.values))


class Uniform(Param):
    def __init__(self, start, end, default=None):
        self.start = start
        self.end = end
        self.set(default)

    def generate(self):
        self.set(random.uniform(self.start, self.end))


class ConfigType(type):
    def __getattribute__(self, name):
        if name.startswith('__'):
            return type.__getattribute__(self, name)
        if name in self.__dict__:
            value = self.__dict__[name]
            if isinstance(value, Param):
                return value.get()
            return value
        raise AttributeError

    def __setattr__(self, name, value):
        if name.startswith('__'):
            return type.__setattr__(self, name, value)
        if name in self.__dict__:
            found = self.__dict__[name]
            if isinstance(found, Param):
                found.set(value)
                return value
            if type(value) != type(self.__dict__[name]):
                value = type(self.__dict__[name])(value)
        return type.__setattr__(self, name, value)

    def __repr__(self):
        return str({key: getattr(self, key) for key in self.__dict__ if not key.startswith('__')})


def generate(config):
    for value in config.__dict__.values():
        if isinstance(value, Param):
            value.generate()

def usage(config):
    print('usage: %s [options]' % sys.argv[0])
    print('options:')
    for key in sorted(filter(lambda k: not k.startswith('__'), config.__dict__)):
        print('  --%s=%s' % (key, str(getattr(config, key))))
    sys.exit(1)

def command_line(config):
    for arg in sys.argv[1:]:
        if arg == '--help' or arg == '-h':
            usage(config)
        elif arg.startswith('--config='):
            filename = arg.split('=', 1)[1]
            if filename.startswith('{'):
                load(config, json.parse(filename))
            else:
                with open(filename) as fp:
                    load(config, json.parse(fp.read()))
        elif arg.startswith('--') and '=' in arg:
            key, value = arg[2:].split('=', 1)
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print('error: unknown argument "%s", try --help' % key)
                sys.exit(1)
        else:
            print('error: malformed argument "%s", try --help' % arg)
            sys.exit(1)

def save(config):
    return {key: getattr(config, key) for key in config.__dict__ if not key.startswith('__')}

def load(config, dictionary):
    for key in config.__dict__:
        if not key.startswith('__') and key in dictionary:
            setattr(config, key, dictionary[key])

class Config:
    __metaclass__ = ConfigType

if __name__ == '__main__':
    class C(Config):
        x = 1
        y = 2
        z = Choice([3, 4, 5], default=3)
        k = Uniform(0, 1, default=0.5)

    generate(C)
    load(C, {'x': 3, 'k': 0.1})
    print(save(C))
    print(C)

    for i in range(10):
        generate(C)
        print(C)

