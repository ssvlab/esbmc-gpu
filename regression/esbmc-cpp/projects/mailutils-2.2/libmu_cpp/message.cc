/*
   GNU Mailutils -- a suite of utilities for electronic mail
   Copyright (C) 2004, 2006, 2007, 2009, 2010 Free Software Foundation,
   Inc.

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

#include <cstdlib>
#include <mailutils/cpp/message.h>

using namespace mailutils;

//
// Message
//

Message :: Message ()
{  
  int status = mu_message_create (&msg, this);
  if (status)
    throw Exception ("Message::Message", status);

  this->owner = true;
}

Message :: Message (const mu_message_t msg)
{
  if (msg == 0)
    throw Exception ("Message::Message", EINVAL);

  this->msg = msg;
  this->owner = false;
}

Message&
Message :: operator = (const Message& m)
{
  if (this != &m)
    {
      if (this->owner)
	mu_message_destroy (&this->msg, this);

      int status = mu_message_create_copy (&this->msg, m.msg);
      if (status)
	throw Exception ("Message::operator=", status);

      this->owner = true;
    }
  return *this;
}

Message :: ~Message ()
{
  if (this->owner)
    mu_message_destroy (&msg, this);
}

Attribute&
Message :: get_attribute ()
{
  mu_attribute_t c_attr;

  int status = mu_message_get_attribute (msg, &c_attr);
  if (status)
    throw Exception ("Message::get_attribute", status);

  return *new Attribute (c_attr);
}

Body&
Message :: get_body ()
{
  mu_body_t c_body;

  int status = mu_message_get_body (msg, &c_body);
  if (status)
    throw Exception ("Message::get_body", status);

  return *new Body (c_body);
}

Envelope&
Message :: get_envelope ()
{
  mu_envelope_t c_env;

  int status = mu_message_get_envelope (msg, &c_env);
  if (status)
    throw Exception ("Message::get_envelope", status);

  return *new Envelope (c_env);
}

Header&
Message :: get_header ()
{
  mu_header_t c_hdr;

  int status = mu_message_get_header (msg, &c_hdr);
  if (status)
    throw Exception ("Message::get_header", status);

  return *new Header (c_hdr);
}

Stream&
Message :: get_stream ()
{
  mu_stream_t c_stream;

  int status = mu_message_get_stream (msg, &c_stream);
  if (status)
    throw Exception ("Message::get_stream", status);

  return *new Stream (c_stream);
}

void
Message :: set_stream (const Stream& stream)
{
  int status = mu_message_set_stream (msg, stream.stm, this);
  if (status)
    throw Exception ("Message::set_stream", status);
}


bool
Message :: is_multipart ()
{
  int pmulti;
  int status = mu_message_is_multipart (msg, &pmulti);
  if (status)
    throw Exception ("Message::is_multipart", status);
  return (bool) pmulti;
}

size_t
Message :: size ()
{
  size_t c_size;
  int status = mu_message_size (msg, &c_size);
  if (status)
    throw Exception ("Message::size", status);
  return c_size;
}

size_t
Message :: lines ()
{
  size_t c_lines;
  int status = mu_message_lines (msg, &c_lines);
  if (status)
    throw Exception ("Message::lines", status);
  return c_lines;
}

size_t
Message :: get_num_parts ()
{
  size_t c_parts;
  int status = mu_message_get_num_parts (msg, &c_parts);
  if (status)
    throw Exception ("Message::get_num_parts", status);
  return c_parts;
}

Message&
Message :: get_part (const size_t npart)
{
  mu_message_t c_part;

  int status = mu_message_get_part (msg, npart, &c_part);
  if (status)
    throw Exception ("Message::get_part", status);

  return *new Message (c_part);
}

void
Message :: save_attachment ()
{
  int status = mu_message_save_attachment (msg, NULL, NULL);
  if (status)
    throw Exception ("Message::save_attachment", status);
}

void
Message :: save_attachment (const std::string& filename)
{
  int status = mu_message_save_attachment (msg, filename.c_str (), NULL);
  if (status)
    throw Exception ("Message::save_attachment", status);
}

Message&
Message :: unencapsulate ()
{
  mu_message_t c_msg;

  int status = mu_message_unencapsulate (msg, &c_msg, NULL);
  if (status)
    throw Exception ("Message::unencapsulate", status);

  return *new Message (c_msg);
}

std::string
Message :: get_attachment_name ()
{
  char *c_name;
  std::string name;

  int status = mu_message_aget_decoded_attachment_name (msg, NULL, &c_name,
							NULL);
  if (status)
    throw Exception ("Message::get_attachment_name", status);
  if (c_name) {
    name = c_name;
    free (c_name);
  }
  return name;
}

std::string
Message :: get_attachment_name (const std::string& charset, char *lang)
{
  char *c_name;
  std::string name;

  int status = mu_message_aget_decoded_attachment_name (msg, charset.c_str (),
							&c_name, &lang);
  if (status)
    throw Exception ("Message::get_attachment_name", status);
  if (c_name) {
    name = c_name;
    free (c_name);
  }
  return name;
}

