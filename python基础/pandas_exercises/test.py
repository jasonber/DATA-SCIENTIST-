# chipotle
import pandas as pd
chipo2 = pd.read_csv('/home/zhangzhiliang/Documents/Book/coding/pandas_exercises/chipotle.tsv', sep = '\t')


# clean the item_price column and transform it in a float
prices2 = [float(value[1 : -1]) for value in chipo2.item_price]

# reassign the column with the cleaned prices
chipo2.item_price = prices2

# delete the duplicates in item_name and quantity
chipo2_filtered = chipo2.drop_duplicates(['item_name','quantity'])

# select only the products with quantity equals to 1
chipo2_one_prod = chipo2_filtered[chipo2_filtered.quantity == 1]

chipo2_one_prod[chipo2_one_prod['item_price']>10].item_name.nunique()