# -*- coding: utf-8 -*-
import scrapy


class Baoxian2Spider(scrapy.Spider):
    name = 'baoxian2'
    allowed_domains = ['douban.com']
    start_urls = ['http://movie.douban.com/top250']
    
    
    
    def parse(self, response):
        # test = response.xpath('//div[@id="wrapper"]/、div[@class="article"]/h2/text()')
        test = response.body
        print(test)
        pass
