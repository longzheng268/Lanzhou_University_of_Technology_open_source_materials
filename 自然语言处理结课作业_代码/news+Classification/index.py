import os
import time
import requests
from lxml import etree
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


def get_label(url):
    if "health" in url:
        return "health"
    elif "society" in url:
        return "society"
    elif "finance" in url:
        return "finance"
    elif "military" in url:
        return "military"
    elif "edu" in url:
        return "education"
    else:
        return "无"


def get_htmls():
    service = Service(executable_path="./chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    for base_url in ["http://health.people.com.cn/GB/408647/", "http://society.people.com.cn/GB/86800/", "http://finance.people.com.cn/GB/414330/", "http://military.people.com.cn/GB/52936/", "http://edu.people.com.cn/GB/367001/"][3:]:
        for index in range(1, 13):
            url = base_url + "index" + str(index) + ".html"
            driver.get(url)
            time.sleep(1)
            driver.maximize_window()
            page_source = driver.page_source
            label = get_label(base_url)
            with open(f"htmls/{label}_index{index}.html", "w", encoding="utf-8") as f:
                f.write(page_source)


def parse_htmls():
    service = Service(executable_path="./chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    with open("data/人民网新闻数据.csv", "w", encoding="utf-8") as ff:
        ff.write("\t".join(["新闻类别", "标题", "发布时间", "链接详情", "新闻内容"]) + "\n")
        for html_name in os.listdir("htmls"):
            html_path = "htmls/" + html_name
            with open(html_path, "r", encoding="utf-8") as f:
                html = etree.HTML(f.read(), etree.HTMLParser())
            if "health" in html_name:
                hrefs1 = html.xpath("//ul/div[@class='newsItems']/a/@href")
                times1 = html.xpath("//ul/div[@class='newsItems']/div[@class='n_time']/text()")
                titles1 = html.xpath("//ul/div[@class='newsItems']/a/text()")
                hrefs2 = html.xpath("//ul/li/div[@class='newsItems']/a/@href")
                times2 = html.xpath("//ul/li/div[@class='newsItems']/div[@class='n_time']/text()")
                titles2 = html.xpath("//ul/li/div[@class='newsItems']/a/text()")
                hrefs, times, titles = [item if ".com.cn" in item else "http://health.people.com.cn" + item for item in hrefs1 + hrefs2], times1 + times2, titles1 + titles2
                label = "health"
            elif "finance" in html_name:
                hrefs = [item if ".com.cn" in item else "http://finance.people.com.cn" + item for item in
                         html.xpath("//div/ul[@class='list_16 mt10']/li/a/@href")]
                times = html.xpath("//div/ul[@class='list_16 mt10']/li/em/text()")
                titles = html.xpath("//div/ul[@class='list_16 mt10']/li/a/text()")
                label = "finance"
            elif "society" in html_name:
                hrefs = [item if ".com.cn" in item else "http://society.people.com.cn" + item for item in
                         html.xpath("//div/ul[@class='list_16 mt10']/li/a/@href")]
                times = html.xpath("//div/ul[@class='list_16 mt10']/li/em/text()")
                titles = html.xpath("//div/ul[@class='list_16 mt10']/li/a/text()")
                label = "society"
            elif "military" in html_name:
                hrefs = [item if ".com.cn" in item else "http://military.people.com.cn" + item for item in
                         html.xpath("//div/ul[@class='list_16 mt10']/li/a/@href")]
                times = html.xpath("//div/ul[@class='list_16 mt10']/li/em/text()")
                titles = html.xpath("//div/ul[@class='list_16 mt10']/li/a/text()")
                label = "military"
            elif "edu" in html_name:
                hrefs = [item if ".com.cn" in item else "http://edu.people.com.cn" + item for item in
                         html.xpath("//div/ul[@class='list_16 mt10']/li/a/@href")]
                times = html.xpath("//div/ul[@class='list_16 mt10']/li/em/text()")
                titles = html.xpath("//div/ul[@class='list_16 mt10']/li/a/text()")
                label = "education"
            else:
                continue

            for i, href in enumerate(hrefs):
                driver.get(href)
                time.sleep(1)
                driver.maximize_window()
                page_source = driver.page_source
                html_news = etree.HTML(page_source, etree.HTMLParser())
                contents = html_news.xpath("//div/p[@style='text-indent: 2em;']/text()")
                for content in contents:
                    ff.write("\t".join([label, str(titles[i]), str(times[i]), str(hrefs[i]), str(content)]) + "\n")


if __name__ == '__main__':
    get_htmls()
    parse_htmls()










