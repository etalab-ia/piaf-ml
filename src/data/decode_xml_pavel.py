from pathlib import Path
import base64

with open(Path('/home/robin/questions_piaf/data/15082020-ServicePublic_QR_20170612_20180612.xml')) as filo:
    lines = [l.strip() for l in filo]
new_list = []
for l in lines:
    if "<incoming>" in l:
        temp = l.replace("<incoming>", "")
        temp = temp.replace("</incoming>", "")
        temp = f"<incoming>{base64.b64decode(temp).decode('utf-8')}</incoming>"
    elif "<outgoing>" in l:
        temp = l.replace("<outgoing>", "")
        temp = temp.replace("</outgoing>", "")
        temp = f"<outgoing>{base64.b64decode(temp).decode('utf-8')}</outgoing>"
    else:
        temp = l
    new_list.append(temp)
with open("/home/pavel/Downloads/15082020-ServicePublic_QR_20170612_20180612_str.xml", "w") as filo:
    for l in new_list:
        filo.write(l)