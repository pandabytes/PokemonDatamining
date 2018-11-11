package edu.ucr.cs235.crawler;

import java.io.IOException;

public class PokemonApplication
{

  public static void main(String[] args) throws IOException, InterruptedException
  {
    PokemonCrawler crawler = new PokemonCrawler();
    Runtime.getRuntime().addShutdownHook(crawler);
    
//    crawler.scrapePokemonTest(1, 19, 19);
//    crawler.scrapePokemonTest(7, 800, 800);
//    crawler.scrapePokemonTest(1, 6, 6);
//    crawler.scrapePokemonTest(1, 24, 24);
//    crawler.scrapePokemonTest(7, 771, 771);
//    crawler.scrapePokemonTest(7, 779, 779);
    
    String html = crawler.getPageFromUrl(PokemonCrawler.PokemonBaseUrl + 
                                         "/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number");
    crawler.scrapePokemonList(html);
    
    
//  HashMap<String, Integer> stats = scrapeStats();
////
//Pokemon pokemon = new Pokemon(generation, pokedex, formName, types, null, 
//                              genders[0], genders[1], height, weight,
//                              category, eggGroups, 0, 0, 0, 0, 0, 0, 0);
  }

}
