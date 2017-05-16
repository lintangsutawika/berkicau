using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace berkicau_rb
{
    public class Program
    {
        public static void Main(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine("Parameter tidak ada");
                Console.WriteLine("Masukkan parameter: berkicau-rb [perintah] [inputfile] [outputfile]");
                Console.WriteLine("  [perintah]: trainer");
                Console.WriteLine("  [inputfile]: inputfile.txt");
                Console.WriteLine("  [outputfile]: (Default: rb-output.txt)");
                Console.ReadKey();
                return;
            }

            if (args[0] == "trainer")
            {
                var inputFile = System.IO.File.ReadLines(args[1]);
                var regexTag = new Regex("<ENAMEX(.*?)>(.*?)</ENAMEX>");
                var regexType = new Regex(@"(?<=\bTYPE="")[^""]*");
                var regexInnerText = new Regex(@"<[^>]*>");

                var nerList = new List<string>();
                var kamusNER = new List<string>();

                var i = 1;

                foreach (var line in inputFile)
                {
                    var matchs = regexTag.Matches(line);

                    foreach (var tagMatch in matchs)
                    {
                        // Jika belum ada 
                        if (nerList.BinarySearch(tagMatch.ToString()) < 0)
                        {
                            nerList.Add(tagMatch.ToString());
                            nerList.Sort(); // di sort langsung agar bisa pakai binary search
                        }


                    }

                    i++;
                }

                // Proses kamus ke OIB Format --> simpan ke outputtext
                foreach (var tagged in nerList)
                {
                    var innerText = regexInnerText.Replace(tagged.ToString(), String.Empty);
                    var nerType = regexType.Match(tagged.ToString());
                    var words = innerText.Split(null);

                    for (var iWord = 0; iWord < words.Length; iWord++)
                    {
                        // Console.WriteLine("  --> " + words[iWord] + " " + (iWord == 0 ? "B-" : "I-") + nerType.ToString());
                        kamusNER.Add(words[iWord] + " " + (iWord == 0 ? "B-" : "I-") + nerType.ToString());
                    }
                }

                //foreach (var kamus in kamusNER)
                //    Console.WriteLine(kamus);

                var outputFile = "rb-model.txt";
                if (args.Length == 3) outputFile = args[2];

                // tulis ke model
                File.WriteAllLines(outputFile, kamusNER);
                Console.WriteLine("Model berhasil di simpan : " + outputFile);

                Console.ReadKey();
                return;
            }
        }
    }
}
