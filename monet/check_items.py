# check_items.py
import json

with open("index/items.json", "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"总条目数: {len(items)}\n")

# 搜索包含"门票"、"票价"、"浪漫与星"的条目
keywords = ["门票", "票价", "优惠", "购票", "浪漫与星", "早鸟票", "亲子票"]

for kw in keywords:
    found = [it for it in items if kw in it["text"] or kw in it["title"]]
    print(f"【{kw}】匹配 {len(found)} 条")
    if found:
        # 显示第一条
        print(f"  示例: {found[0]['title'][:40]}...")
        print(f"  来源: {found[0]['source']}")
        print(f"  分类: {found[0]['category']}")
        print()