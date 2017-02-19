/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2009, 2010 Free Software Foundation, Inc.

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General
   Public License along with this library; if not, write to the
   Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301 USA
*/

#include <mailutils/cpp/folder.h>

using namespace mailutils;

//
// Folder
//

Folder :: Folder (const std::string& name)
{
  int status = mu_folder_create (&folder, name.c_str ());
  if (status)
    throw Exception ("Folder::Folder", status);
}

Folder :: Folder (const mu_folder_t folder)
{
  if (folder == 0)
    throw Exception ("Folder::Folder", EINVAL);

  this->folder = folder;
}

Folder&
Folder :: operator = (const Folder& f)
{
  if (this != &f)
    {
      if (this->folder)
	mu_folder_destroy (&this->folder);
      this->folder = f.folder;
    }
  return *this;
}

Folder :: ~Folder ()
{
  mu_folder_destroy (&folder);
}


void
Folder :: open ()
{
  int status = mu_folder_open (folder, MU_STREAM_READ);
  if (status)
    throw Exception ("Folder::open", status);
}

void
Folder :: open (int flag)
{
  int status = mu_folder_open (folder, flag);
  if (status)
    throw Exception ("Folder::open", status);
}

void
Folder :: close ()
{
  int status = mu_folder_close (folder);
  if (status)
    throw Exception ("Folder::close", status);
}

List&
Folder :: list (const std::string& dirname, void* pattern,
		size_t max_level = 0)
{
  mu_list_t c_list;

  int status = mu_folder_list (folder, dirname.c_str (), pattern,
			       max_level, &c_list);
  if (status)
    throw Exception ("Folder::list", status);

  return *new List (c_list);
}

List&
Folder :: enumerate (const std::string& name, void* pattern,
		     int flags, size_t max_level,
		     mu_folder_enumerate_fp enumfun, void* enumdata)
{
  mu_list_t c_list;

  int status = mu_folder_enumerate (folder, name.c_str (), pattern,
				    flags, max_level, &c_list,
				    enumfun, enumdata);
  if (status)
    throw Exception ("Folder::enumerate", status);

  return *new List (c_list);
}

Stream&
Folder :: get_stream ()
{
  mu_stream_t c_stream;

  int status = mu_folder_get_stream (folder, &c_stream);
  if (status)
    throw Exception ("Folder::get_stream", status);

  return *new Stream (c_stream);
}

void
Folder :: set_stream (const Stream& stream)
{
  int status = mu_folder_set_stream (folder, stream.stm);
  if (status)
    throw Exception ("Folder::set_stream", status);
}

Url&
Folder :: get_url ()
{
  mu_url_t c_url;

  int status = mu_folder_get_url (folder, &c_url);
  if (status)
    throw Exception ("Folder::get_url", status);

  return *new Url (c_url);
}

