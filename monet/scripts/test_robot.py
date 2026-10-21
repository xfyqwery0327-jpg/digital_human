import sys
import json
import requests

URL = "http://10.0.7.232:8848/api/robot/ask"
QUESTIONS = [
    "",
    "莫奈介绍下自己吧",
    "介绍下你的作品吧",
    "睡莲是在哪里画的？"
]

def main():
    if len(sys.argv) < 2:
        print("用法：python test_robot.py 1|2|3")
        return
    idx = int(sys.argv[1])
    if idx not in (1,2,3):
        print("只能输入1、2、3")
        return
    query = QUESTIONS[idx]
    body = {
        "robot_id": "script-01",
        "session_id": "robot_1789117605_d8cbaabe",
        "query": query
    }
    print(f"query={query}")
    resp = requests.post(URL, json=body, timeout=30)
    print(f"status={resp.status_code}")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
