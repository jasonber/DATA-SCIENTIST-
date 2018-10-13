# Pokemon
import pandas as pd

# 创建一个字典
raw_data = {'name': ['Bulbasaru', 'Charmander', 'Squirtle', 'Caterpie'],
            'evolution': ['Ivysaru', 'Charmenleon', 'Wartotrtle', 'Metapod'],
            'type': ['grass', 'fire', 'water', 'bug'],
            'pokedex': ['yes', 'no', 'yes', 'no'],
            'hp': [45, 39, 44, 45]
            }

# 命名数据框
pokemon = pd.DataFrame(raw_data)

# 调整数据框列名的顺序为name, type, hp, evolution, pokedex
pokemon = pokemon[['name', 'type', 'hp', 'evolution', 'pokedex']]

# 插入地区属性
pokemon['place'] = ['park', 'street', 'lake', 'forest']

# 输出各列的数据类型
pokemon.dtypes
