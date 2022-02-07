import sqlite3

# Connect till fil som ska vara tom
connection_object = sqlite3.connect("Dailytask.db")
# skapa ett cursor objekt --> Sammanlänkar connection med db-fil ännu mer!
cursor_object = connection_object.cursor()

#cursor_object.execute("create table daily_tasks (id integer, name task)")

cursor_object.execute("insert into daily_tasks values (0, 'master thesis meeting')")

connection_object.commit()