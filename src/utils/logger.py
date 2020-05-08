import datetime

def log(text:str, title=False):
    now = datetime.datetime.now()
    t = now.strftime("%Y-%m-%d %H:%M:%S")
    if title:
        print(f"[{t}] ##### {text} #####")
    else:
        print(f"[{t}] {text}")
