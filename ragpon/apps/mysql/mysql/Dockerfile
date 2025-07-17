# For Linux:
    # podman machine ssh
    # cd /home/user/ragpon/ragpon/apps/mysql/mysql
    # podman build -t ragpon-mysql .

FROM docker.io/library/mysql:8.0

COPY my.cnf /etc/mysql/my.cnf
RUN chmod 644 /etc/mysql/my.cnf
RUN microdnf update -y
RUN microdnf install -y bash gettext

COPY init.sql.template /tmp/init.sql.template

ENTRYPOINT ["/bin/sh", "-c", "\
    envsubst < /tmp/init.sql.template > /docker-entrypoint-initdb.d/init.sql && \
    exec docker-entrypoint.sh \"$@\" \
", "--"]

CMD ["mysqld"]

# how to build:
# cd ragpon
# cd /home/user/ragpon/ragpon/apps/mysql/mysql
# podman build -t ragpon-mysql:latest .

# how to run:
# see how_to_run_memo.txt

