# For Linux:
    # podman machine ssh
    # cd /home/user/ragpon/ragpon/apps/mysql/mysql
    # podman build -t ragpon-mysql .

FROM docker.io/library/mysql:8.0

COPY my.cnf /etc/mysql/my.cnf
RUN chmod 644 /etc/mysql/my.cnf

# Copy the template and the init script into initdb.d directory
COPY init.sql.template /docker-entrypoint-initdb.d/init.sql.template
COPY 00_generate_init_sql.sh /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/00_generate_init_sql.sh

# # Substitute all placeholders at container start-up, then exec the official entrypoint
# ENTRYPOINT ["/bin/sh", "-c", "\
#     sed \
#       -e \"s|\\${RW_USER}|$RW_USER|g\" \
#       -e \"s|\\${RW_HOST}|$RW_HOST|g\" \
#       -e \"s|\\${RW_PASSWORD}|$RW_PASSWORD|g\" \
#       -e \"s|\\${MYSQL_DATABASE}|$MYSQL_DATABASE|g\" \
#       /tmp/init.sql.template > /docker-entrypoint-initdb.d/init.sql && \
#     exec docker-entrypoint.sh \"$@\" \
# "]

# CMD ["mysqld"]

# how to build:
# cd ragpon
# cd /home/user/ragpon/ragpon/apps/mysql/mysql
# podman build -t ragpon-mysql:latest .

# how to run:
# see how_to_run_memo.txt

