using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace berkicau_rb
{
    public class Program
    {
        public static void Main(string[] args)
        {
            if (args.Count() < 2)
            {
                Console.WriteLine("Parameter tidak ada");
                Console.WriteLine("Masukkan parameter: ner-rule-based [perintah] [inputfile] [outputfile]");
                Console.WriteLine("  [perintah]: trainer");
                Console.WriteLine("  [inputfile]: inputfile.txt");
                Console.WriteLine("  [outputfile]: (Default: output.txt)");
                Console.ReadKey();
                return;
            }

            if (args[0] == "trainer")
            {
                var inputFile = System.IO.File.ReadLines(args[1]);
                var regexTag = new Regex("<ENAMEX(.*?)>(.*?)</ENAMEX>");
                var regexType = new Regex(@"(?<=\bTYPE="")[^""]*");

                var i = 1;

                foreach (var line in inputFile)
                {
                    var matchs = regexTag.Matches(line);

                    if (matchs.Count > 0)
                        Console.WriteLine("Line " + i + ":");

                    foreach (var tagMatch in matchs)
                    {
                        var innerText = Regex.Replace(tagMatch.ToString(), @"<[^>]*>", String.Empty);
                        var nerType = regexType.Match(tagMatch.ToString());

                        Console.WriteLine("  " + tagMatch.ToString());
                        Console.WriteLine("  --> " + innerText + " " + nerType.ToString());
                    }

                    i++;
                }

                Console.ReadKey();
                return;
            }
        }
    }
}
