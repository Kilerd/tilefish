// A small JSON reader for the C++ gate (test code only: the library and its headers take no JSON dependency). It reads what
// python/tests/rust_gate.py writes: objects, arrays, strings without unusual escapes, integers, floats, true, false and null.
#ifndef TILEFISH_GATE_JSON_HPP
#define TILEFISH_GATE_JSON_HPP

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace json {

class Value {
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Value() = default;
    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::Null; }

    const Value& operator[](const std::string& key) const {
        static const Value null;
        if (kind_ != Kind::Object) return null;
        for (const auto& kv : object_) {
            if (kv.first == key) return kv.second;
        }
        return null;
    }
    const Value& operator[](size_t i) const { return array_.at(i); }
    bool has(const std::string& key) const {
        if (kind_ != Kind::Object) return false;
        for (const auto& kv : object_) {
            if (kv.first == key) return true;
        }
        return false;
    }
    size_t size() const { return kind_ == Kind::Array ? array_.size() : object_.size(); }
    const std::vector<Value>& array() const { return array_; }
    const std::vector<std::pair<std::string, Value>>& object() const { return object_; }
    const std::string& str() const {
        if (kind_ != Kind::String) throw std::runtime_error("json: not a string");
        return string_;
    }
    int64_t i64() const {
        if (kind_ != Kind::Number) throw std::runtime_error("json: not a number");
        return integer_ ? int_ : static_cast<int64_t>(double_);
    }
    uint64_t u64() const { return static_cast<uint64_t>(i64()); }
    double f64() const { return integer_ ? static_cast<double>(int_) : double_; }
    bool boolean() const { return kind_ == Kind::Bool && bool_; }
    bool operator==(const std::string& s) const { return kind_ == Kind::String && string_ == s; }
    bool operator!=(const std::string& s) const { return !(*this == s); }

private:
    friend class Parser;
    Kind kind_ = Kind::Null;
    bool bool_ = false;
    bool integer_ = true;
    int64_t int_ = 0;
    double double_ = 0;
    std::string string_;
    std::vector<Value> array_;
    std::vector<std::pair<std::string, Value>> object_;
};

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Value parse() {
        Value v = value();
        ws();
        if (i_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("json: ") + what + " at " + std::to_string(i_)); }
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) ++i_;
    }
    char peek() {
        ws();
        if (i_ >= s_.size()) fail("unexpected end");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        ++i_;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (i_ >= s_.size()) fail("unterminated string");
            char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                char e = s_[i_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        unsigned code = std::stoul(s_.substr(i_, 4), nullptr, 16);
                        i_ += 4;
                        if (code < 0x80) {
                            out += static_cast<char>(code);
                        } else if (code < 0x800) {
                            out += static_cast<char>(0xC0 | (code >> 6));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (code >> 12));
                            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                out += c;
            }
        }
        return out;
    }
    Value value() {
        Value v;
        char c = peek();
        if (c == '{') {
            ++i_;
            v.kind_ = Value::Kind::Object;
            if (peek() == '}') {
                ++i_;
                return v;
            }
            while (true) {
                std::string key = string();
                expect(':');
                v.object_.emplace_back(std::move(key), value());
                if (peek() == ',') {
                    ++i_;
                    continue;
                }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            ++i_;
            v.kind_ = Value::Kind::Array;
            if (peek() == ']') {
                ++i_;
                return v;
            }
            while (true) {
                v.array_.push_back(value());
                if (peek() == ',') {
                    ++i_;
                    continue;
                }
                expect(']');
                return v;
            }
        }
        if (c == '"') {
            v.kind_ = Value::Kind::String;
            v.string_ = string();
            return v;
        }
        if (s_.compare(i_, 4, "true") == 0) {
            i_ += 4;
            v.kind_ = Value::Kind::Bool;
            v.bool_ = true;
            return v;
        }
        if (s_.compare(i_, 5, "false") == 0) {
            i_ += 5;
            v.kind_ = Value::Kind::Bool;
            return v;
        }
        if (s_.compare(i_, 4, "null") == 0) {
            i_ += 4;
            return v;
        }
        size_t start = i_;
        bool integer = true;
        while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '-' || s_[i_] == '+' || s_[i_] == '.' ||
                                  s_[i_] == 'e' || s_[i_] == 'E')) {
            if (s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E') integer = false;
            ++i_;
        }
        if (start == i_) fail("unexpected character");
        v.kind_ = Value::Kind::Number;
        std::string text = s_.substr(start, i_ - start);
        v.integer_ = integer;
        if (integer) {
            v.int_ = std::stoll(text);
        } else {
            v.double_ = std::stod(text);
        }
        return v;
    }

    const std::string& s_;
    size_t i_ = 0;
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

inline Value read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parse(ss.str());
}

}  // namespace json

#endif  // TILEFISH_GATE_JSON_HPP
