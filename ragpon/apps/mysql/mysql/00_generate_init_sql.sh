set -e

echo "[INIT] Expanding template /docker-entrypoint-initdb.d/init.sql.template"

sed \
  -e "s|\\${RW_USER}|$RW_USER|g" \
  -e "s|\\${RW_HOST}|$RW_HOST|g" \
  -e "s|\\${RW_PASSWORD}|$RW_PASSWORD|g" \
  -e "s|\\${MYSQL_DATABASE}|$MYSQL_DATABASE|g" \
  /docker-entrypoint-initdb.d/init.sql.template > /docker-entrypoint-initdb.d/10_init.sql

echo "[INIT] Generated 10_init.sql:"
cat /docker-entrypoint-initdb.d/10_init.sql