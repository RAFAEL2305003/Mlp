#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace json
{
	enum class kind
	{
		null,
		boolean,
		number,
		string,
		array,
		object
	};

	class value;
	using array = std::vector<value>;
	using object = std::map<std::string, value>;

	class value
	{
		public:
			kind type;
			bool b;
			double n;
			std::string s;
			std::shared_ptr<array> arr;
			std::shared_ptr<object> obj;

			value() : type(kind::null), b(false), n(0.0) {}

			bool as_bool() const
			{
				assert(type == kind::boolean);
				return b;
			}

			double as_number() const
			{
				assert(type == kind::number);
				return n;
			}

			long as_int() const
			{
				assert(type == kind::number);
				return static_cast<long>(n);
			}

			const std::string& as_string() const
			{
				assert(type == kind::string);
				return s;
			}

			const array& as_array() const
			{
				assert(type == kind::array);
				return *arr;
			}

			const object& as_object() const
			{
				assert(type == kind::object);
				return *obj;
			}

			const value& at(const std::string& key) const
			{
				assert(type == kind::object);
				auto it = obj->find(key);
				if(it == obj->end())
				{
					throw std::runtime_error("json: missing key '" + key + "'");
				}
				return it->second;
			}

			bool has(const std::string& key) const
			{
				return type == kind::object && obj->find(key) != obj->end();
			}
	};

	class parser
	{
		private:
			const std::string& src;
			std::size_t pos;

			void skip_ws()
			{
				while(pos < src.size())
				{
					char c = src[pos];
					if(c == ' ' || c == '\t' || c == '\n' || c == '\r')
					{
						pos++;
					}
					else
					{
						break;
					}
				}
			}

			char peek()
			{
				skip_ws();
				if(pos >= src.size())
				{
					throw std::runtime_error("json: unexpected end of input");
				}
				return src[pos];
			}

			void expect(char c)
			{
				skip_ws();
				if(pos >= src.size() || src[pos] != c)
				{
					throw std::runtime_error(std::string("json: expected '") + c + "'");
				}
				pos++;
			}

			std::string parse_string()
			{
				expect('"');
				std::string out;
				while(pos < src.size())
				{
					char c = src[pos++];
					if(c == '"')
					{
						return out;
					}
					if(c == '\\' && pos < src.size())
					{
						char e = src[pos++];
						switch(e)
						{
							case '"':  out.push_back('"'); break;
							case '\\': out.push_back('\\'); break;
							case '/':  out.push_back('/'); break;
							case 'n':  out.push_back('\n'); break;
							case 't':  out.push_back('\t'); break;
							case 'r':  out.push_back('\r'); break;
							case 'b':  out.push_back('\b'); break;
							case 'f':  out.push_back('\f'); break;
							default:   out.push_back(e); break;
						}
					}
					else
					{
						out.push_back(c);
					}
				}
				throw std::runtime_error("json: unterminated string");
			}

			value parse_number()
			{
				std::size_t start = pos;
				if(src[pos] == '-' || src[pos] == '+') pos++;
				while(pos < src.size())
				{
					char c = src[pos];
					if((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
					{
						pos++;
					}
					else
					{
						break;
					}
				}
				value v;
				v.type = kind::number;
				v.n = std::strtod(src.c_str() + start, nullptr);
				return v;
			}

			value parse_literal(const std::string& word, value v)
			{
				if(src.compare(pos, word.size(), word) != 0)
				{
					throw std::runtime_error("json: invalid literal");
				}
				pos += word.size();
				return v;
			}

			value parse_array()
			{
				expect('[');
				value v;
				v.type = kind::array;
				v.arr = std::make_shared<array>();
				skip_ws();
				if(pos < src.size() && src[pos] == ']')
				{
					pos++;
					return v;
				}
				while(true)
				{
					v.arr->push_back(parse_value());
					skip_ws();
					if(pos < src.size() && src[pos] == ',')
					{
						pos++;
						continue;
					}
					expect(']');
					return v;
				}
			}

			value parse_object()
			{
				expect('{');
				value v;
				v.type = kind::object;
				v.obj = std::make_shared<object>();
				skip_ws();
				if(pos < src.size() && src[pos] == '}')
				{
					pos++;
					return v;
				}
				while(true)
				{
					skip_ws();
					std::string key = parse_string();
					expect(':');
					(*v.obj)[key] = parse_value();
					skip_ws();
					if(pos < src.size() && src[pos] == ',')
					{
						pos++;
						continue;
					}
					expect('}');
					return v;
				}
			}

		public:
			parser(const std::string& s) : src(s), pos(0) {}

			value parse_value()
			{
				char c = peek();
				if(c == '{') return parse_object();
				if(c == '[') return parse_array();
				if(c == '"')
				{
					value v;
					v.type = kind::string;
					v.s = parse_string();
					return v;
				}
				if(c == 't')
				{
					value v;
					v.type = kind::boolean;
					v.b = true;
					return parse_literal("true", v);
				}
				if(c == 'f')
				{
					value v;
					v.type = kind::boolean;
					v.b = false;
					return parse_literal("false", v);
				}
				if(c == 'n')
				{
					value v;
					v.type = kind::null;
					return parse_literal("null", v);
				}
				return parse_number();
			}
	};

	value parse(const std::string& src)
	{
		parser p(src);
		return p.parse_value();
	}

	value load(const std::string& filename)
	{
		std::ifstream in(filename);
		if(!in.is_open())
		{
			throw std::runtime_error("json: cannot open file '" + filename + "'");
		}
		std::stringstream ss;
		ss << in.rdbuf();
		return parse(ss.str());
	}
}
