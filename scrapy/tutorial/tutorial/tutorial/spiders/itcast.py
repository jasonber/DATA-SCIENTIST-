# -*- coding: utf-8 -*-
import scrapy


class ItcastSpider(scrapy.Spider):
    name = 'itcast'    # 爬虫名
    allowed_domains = ['itcast.cn']    # 允许爬取的范围
    start_urls = ['http://www.itcast.cn/channel/teacher.shtml']    # 开始的URL地址

    def parse(self, response):
        """
        Args:
             response: 对start_urls链接的所有相应内容
                       可以采用xpath的方法，来进行操作，主要选定抓取的内容
        Yield：
            用于结果的传输，yield也可以节省内存
        Notes:
            parse名字不能改动
        """
        # 某一个xpath的内容
        # teacher = response.xpath(
        #     '//div[@class="tea_con"]//div[@class="li_txt"]/h3/text()').extract()
        # print(teacher)
        
        # 分组
        # 不到具体值的位置，就无法保存值
        teachers = response.xpath(
            "//div[@class='tea_con']//li" # /后面必须要有意义
        )

        for i in teachers:
            item = {}
            item['name'] = i.xpath(".//h3/text()").extract()    # teachers已经是xpath了，而不是一个string
            item['title'] = i.xpath(".//h4/text()").extract()
            print(i)
            print(item)