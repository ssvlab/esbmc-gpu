#! /bin/sh
# This file is part of GNU Mailutils
# Copyright (C) 2004, 2010 Free Software Foundation, Inc.
#
# Written by Sergey Poznyakoff
#
# This file is free software; as a special exception the author gives
# unlimited permission to copy and/or distribute it, with or without
# modifications, as long as this notice is preserved.
#
# GNU Mailutils is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY, to the extent permitted by law; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

cat - <<EOT
/* This file is generated automatically. Please do not edit */
EOT

for module
do
	echo "extern mu_sql_dispatch_t ${module}_dispatch_tab;"
done

cat - <<EOT
static mu_sql_dispatch_t *static_dispatch_tab[] = {
        NULL,
EOT

for module
do
	echo "	&${module}_dispatch_tab,"
done

echo "};"
echo '/* EOF */'
