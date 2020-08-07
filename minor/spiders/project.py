# -*- coding: utf-8 -*-
import scrapy
from ..items import MinorItem

class ProjectSpider(scrapy.Spider):
    name = 'project'
    start_urls = [
        'https://indianexpress.com/section/india/',
        'https://indianexpress.com/section/lifestyle/',
        'https://indianexpress.com/section/sports/'
    ]
    page_number = 2

    def parse(self, response):
        items = MinorItem()
        news = response.css('div.articles')
        for div in news:
            link = div.css('h2.title a::attr(href)').get()
            headline = div.css('h2.title a::text').get()
            date_time = div.css('div.date::text').get()
            ind1 = link.find('article/')+len('article/')
            ind2 = link[ind1:].find('/')
            category = link[ind1:ind1+ind2]
            if category and link and date_time and headline:
                items['headline'] = headline
                items['category'] = category
                items['link'] = link
                items['date_time'] = date_time

                yield items

        next_page = response.css('.next::attr(href)').get()

        if ProjectSpider.page_number <= 30:
            ProjectSpider.page_number += 1
            yield response.follow(next_page, callback=self.parse)