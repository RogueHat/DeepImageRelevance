import sqlite3

with sqlite3.connect('deeprelevance.db', timeout=60*5) as conn:
	c = conn.cursor()
	c.execute('SELECT lastupdate, COUNT(lastupdate) FROM features GROUP BY lastupdate ORDER BY lastupdate')
	tups = c.fetchall()
for t in tups:
	print(t)
