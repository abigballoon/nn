# -*- coding: utf8 -*-
"""
到澎湃新闻那下些语料
"""

import sys
import codecs
import time
import requests
from bs4 import BeautifulSoup

THE_PAPER = "http://www.thepaper.cn/newsDetail_forward_%d"
INVALID_TEXT = u"此文章已下线"


def get_news_text(newsid):
    print "requesting %s"%(THE_PAPER%newsid)
    response = requests.get(THE_PAPER%newsid)
    html = response.text
    soup = BeautifulSoup(html)
    text = soup.find("div", "news_txt")
    time.sleep(0.8)
    return text.text if text and text.text != INVALID_TEXT else ''

def log_task(content):
    with open("task", "w+") as task:
        task.write(content)

def get_task():
    with open("task") as task:
        data = task.read()
    if data:
        return int(data) + 1
    else:
        return 1900000

def download(output, start=0, end=3000000, stop_at=100):
    error = 0
    if not start:
        start = get_task()
    with codecs.open(output, "a+", encoding="utf8") as f:
        while (end and start <= end) and error < stop_at:
            text = get_news_text(start)
            if text:
                f.write(text)
                f.write('\n')
                f.flush()
                error = 0
            else:
                error += 1
            log_task(str(start))
            start += 1

download("news.txt")