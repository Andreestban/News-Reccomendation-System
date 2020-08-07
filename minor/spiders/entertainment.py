# -*- coding: utf-8 -*-
import scrapy
from ..items import MinorItem

class EntertainmentSpider(scrapy.Spider):
    name = 'entertainment'
    start_urls = [
        'https://indianexpress.com/section/entertainment/'
    ]
    page_number = 2

    def parse(self, response):
        items = MinorItem()
        news = response.css('div.articles')
        for div in news:
            link = div.css('div.title a::attr(href)').get()
            headline = div.css('div.title a::text').get()
            date_time = div.css('div.date::text').get()
            category = 'entertainment'
            if date_time and category and link and headline:
                items['headline'] = headline
                items['category'] = category
                items['link'] = link
                items['date_time'] = date_time

                yield items

        next_page = response.css('.next::attr(href)').get()
        print(next_page)
        if EntertainmentSpider.page_number <= 10:
            print(EntertainmentSpider.page_number)
            EntertainmentSpider.page_number += 1
            yield response.follow(next_page, callback = self.parse)