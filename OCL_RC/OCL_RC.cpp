#include <fstream>
#include <iostream>
#include <filesystem>
#include <regex>
#include <sstream>
#include <sys/stat.h>

namespace fs = std::filesystem;

inline bool fileExists(const std::string& name) {
	std::ifstream f(name.c_str());
	return f.good();
}

inline void ReplaceStringInPlace(std::string& subject, const std::string& search,
	const std::string& replace) {
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
}

int main(int argc, char *argv[])
{
	if (argc == 0)
	{
		std::cout << "No params found\n";
		return 0;
	}

	struct stat info;

	if(stat(argv[argc - 1], &info) != 0 || !(info.st_mode & S_IFDIR))
	{
		std::cout << "Include dir not valid:\"" << argv[argc - 1] << "\"\n";
		return 0;
	}

	std::fstream file;
	file.open(std::string(argv[argc - 1]) + "/OCLRes.h", std::fstream::out);
	file << "#pragma once\n";

	if(argc > 1 && (stat(argv[argc - 2], &info) != 0 || !(info.st_mode & S_IFDIR)))
	{
		std::cout << "OCL Source dir not valid:\"" << argv[argc - 2] << "\"\n";
		file.close();
		return 0;
	}

	file << "#include <string>\n";
	std::string basePath = std::string(argv[argc - 2])+"/";

	for (auto& p : fs::directory_iterator(argv[argc - 2]))
	{
		if (p.is_directory())
			continue;
		std::wstring extw = p.path().extension().c_str();
		std::string ext(extw.begin(), extw.end());
		if (ext != ".cl")
			continue;
		std::wstring filenamew = p.path().filename().c_str();
		std::string filename(filenamew.begin(), filenamew.end());

		std::cout << "Processing: " << filename << '\n';

		std::ifstream t(p.path());
		std::stringstream buffer;
		buffer << t.rdbuf();
		std::string source = buffer.str();

		std::smatch m;
		std::regex e("#include\\s*\"[\\w\\/\\\\.]{2,}\"");
		std::regex e2("\"[\\w\\/\\\\.]+\"");
		while (std::regex_search(source, m, e)) {
			for (auto x : m)
			{
				std::string include_line = x.str();
				std::smatch m2;
				std::regex_search(include_line, m2, e2);
				for (std::ssub_match x2 : m2)
				{
					std::string path = x2.str();
					path = path.substr(1, path.length() - 2);
					if (fileExists(basePath + path))
						path = basePath + path;
					else
					{
						path = basePath + "include/" + path;
						if (!fileExists(path))
						{
							std::printf(" ERROR: Could not find cl include file: %s!\n", path.c_str());
							exit(0);
						}
					}

					std::ifstream t2(path);
					std::stringstream buffer2;
					buffer2 << t2.rdbuf();
					ReplaceStringInPlace(source, include_line, buffer2.str() + "\n");
					break;
				}

			}
		};

		ReplaceStringInPlace(source, "\r\n", "\n");
		ReplaceStringInPlace(source, "\n", "\\n");
		ReplaceStringInPlace(source, "\"", "\\\"");
		file << "inline std::string get_" << filename.substr(0, filename.size() - 3) << "() {\n\tstd::string ret;\n";
		size_t i = 0;
		size_t chunk = 100;
		if (chunk < source.size())
		{
			while (i + chunk < source.size())
			{
				if (source[i + chunk - 1] == '\\')
				{
					int offset = 2;
					while (source[i + chunk + offset - 1] == '\\')
						offset += 2;
					file << "\tret += \"" << source.substr(i, chunk + offset) << "\";\n";
					i += chunk + offset;
				}
				else
				{
					std::string chunkstr = source.substr(i, chunk);
					file << "\tret += \"" << chunkstr << "\";\n";
					i += chunk;
				}
			}
			if (i < source.size())
				file << "\tret += \"" << source.substr(i, source.size() - i) << "\";\n";
		}
		else
		{
			file << "\tret += \"" << source << "\";\n";
		}
		file << "\treturn ret;\n}\n\n";
		file.flush();
	}
	file.close();
}