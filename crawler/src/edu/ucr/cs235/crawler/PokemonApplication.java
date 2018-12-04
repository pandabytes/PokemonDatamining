package edu.ucr.cs235.crawler;

import java.io.IOException;

public class PokemonApplication
{
  /*
   * Main function of the crawler program
   */
  public static void main(String[] args) throws IOException, InterruptedException
  {
    PokemonCrawler crawler = new PokemonCrawler();
    Runtime.getRuntime().addShutdownHook(crawler);
        
    String html = crawler.getPageFromUrl(PokemonCrawler.PokemonBaseUrl + 
                                         "/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number");
    crawler.scrapePokemons(html);
  }
}
