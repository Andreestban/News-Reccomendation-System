# -*- coding: utf-8 -*-
import scrapy
from ..items import MinorItem

class TechSpider(scrapy.Spider):
    name = 'tech'
    start_urls = [
        'https://gadgets.ndtv.com/news'
    ]
    page_number = 2

    def parse(self, response):
        items = MinorItem()
        li = response.css('li')
        for news in li:
            link = news.css('div.caption_box a::attr(href)').get()
            headline = news.css('div.caption_box a span.news_listing::text').get()
            date_time = news.css('div.caption_box div.dateline::text').get()
            if date_time:
                ind = date_time.find(',')
                date_time = date_time[ind+1:]
            category = 'technology'
            if headline and date_time and link and category:
                items['headline'] = headline
                items['category'] = category
                items['link'] = link
                items['date_time'] = date_time

                yield items

        next_page = 'https://gadgets.ndtv.com/news/page-' + str(TechSpider.page_number)
        if TechSpider.page_number <= 10:
            print(TechSpider.page_number)
            TechSpider.page_number += 1
            yield response.follow(next_page, callback=self.parse)
