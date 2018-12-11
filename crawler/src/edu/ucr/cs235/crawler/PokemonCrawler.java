package edu.ucr.cs235.crawler;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.select.Elements;

public class PokemonCrawler extends Thread
{
  public final static String PokemonBaseUrl = "https://bulbapedia.bulbagarden.net/wiki";
    public final static String OutputFile = "output" + File.separator + "pokemon.txt";
  private final static int TimeDelayMs = 5500;
  
  private FileWriter m_fileWriter;
  private Document m_document;
  
  /*
   * Hooker function for the JVM. Get call when the program is interrupted (Ctrl+c)
   * Simply close the output file if it's still opened.
   * (non-Javadoc)
   * @see java.lang.Thread#run()
   */
  public void run()
  {
    closeFile();
  }
  
  /**
   * Scrape all Pokemons. Responsible for fetching each Pokemon web page
   * and extract the data from each page.
   * @param html
   * @throws InterruptedException
   * @throws IOException
   */
  public void scrapePokemons(String html) throws InterruptedException, IOException
  {
    // Create the output file and open it in append mode
    createOutputFile();
    openFile(true);
    
    Document document = Jsoup.parse(html);
    HashMap<Integer, Element> pokemonTables = getPokemonTables(document);
    HashSet<String> scrapedPokemons = new HashSet<>();
    
    for (Map.Entry<Integer, Element> entry : pokemonTables.entrySet())
    {
      int generation = entry.getKey();
      Element genTableElement = entry.getValue();

      // Ignore the first row in each table
      for (int j = 1; j < genTableElement.children().size(); j++)
      {
        int nationalPokedex = Integer.parseInt(genTableElement.child(j).child(1).html().substring(1));
        String name = correctPokemonName(genTableElement.child(j).child(3).text());
        
        String url = PokemonBaseUrl + "/" + name + "_(Pokémon)";
        try
        {
          if (!scrapedPokemons.contains(name))
          {
            String pokemonPage = getPageFromUrl(url);

            // A Pokemon may have various forms. Treat these forms as a separate Pokemon
            List<Pokemon> pokemonList = scrapePokemon(pokemonPage, generation, nationalPokedex, name);
            
            // Saved the Pokemon name that we've scraped
            scrapedPokemons.add(name);

            // Write each Pokemon and its forms to file
            for (Pokemon pokemon : pokemonList)
            {
              System.out.println(pokemon.toString());
              m_fileWriter.write(pokemon.toString() + "\n");
            }
          }
        } 
        catch (IOException ex)
        {
          FileWriter ps = new FileWriter("output\\error.txt", true);
          String errorMsg = name + " is not a valid Pokemon name. " + url;
          System.err.println(errorMsg);
          ps.write(errorMsg + "\n");
          ps.close();
        }
        catch (Exception ex)
        {
          FileWriter ps = new FileWriter("output\\error.txt", true);
          ps.write(ex.getMessage() + "\n");
          ex.printStackTrace();
          ps.close();
        }
        Thread.sleep(TimeDelayMs);
      }
    }
    closeFile();
  }
  
  /**
   * This method is used for testing scraping a single Pokemon
   * @param gen
   * @param start
   * @param end
   * @throws IOException
   */
  public void scrapePokemonTest(int gen, int start, int end) throws IOException
  {
    String html = getPageFromUrl(PokemonCrawler.PokemonBaseUrl + 
                  "/List_of_Pok%C3%A9mon_by_National_Pok%C3%A9dex_number");
    Document document = Jsoup.parse(html);
    HashMap<Integer, Element> pokemonTables = getPokemonTables(document);
    Element genTableElement = pokemonTables.get(gen);
    HashSet<String> scrapedPokemon = new HashSet<>();
    
    for (int i = 1; i < genTableElement.children().size(); i++)
    {
      int nationalPokedex = Integer.parseInt(genTableElement.child(i).child(1).html().substring(1));
      String name = correctPokemonName(genTableElement.child(i).child(3).text());
      
      if (!scrapedPokemon.contains(name) && start <= nationalPokedex && nationalPokedex <= end)
      {
        System.out.print("Index: " + i + " | ");
        String url = PokemonBaseUrl + "/" + name + "_(Pokémon)";
        String pokemonPage = getPageFromUrl(url);
        List<Pokemon> pokemonList = scrapePokemon(pokemonPage, gen, nationalPokedex, name);
        pokemonList.forEach(p -> System.out.println(p.toString()));
      }
      scrapedPokemon.add(name);
    }
  }
  
  /**
   * Get all the Pokemon tables from Bulbapedia. Each table
   * represents a generation.
   * @param document
   * @return
   */
  private HashMap<Integer, Element> getPokemonTables(Document document)
  {
    HashMap<Integer, Element> pokemonTables = new HashMap<>();
    Elements elements = document.getElementsByTag("table");
    
    for (Element element : elements)
    {
      // Find tables that contain Pokemon
      if (element.attributes().size() == 2 && element.hasAttr("style") && element.hasAttr("align"))
      {
        int prevIndex = element.elementSiblingIndex() - 1;
        String generationText = element.parent().child(prevIndex).text();
        
        // Only look at tables with the official Generation Pokemon text
        if (!generationText.contains("Generation"))
        {
          continue;
        }
        
        int generation = RomanNumeral.convertToNumber(generationText.split(" ")[1]);
        Element tableElement = element.child(0);
        pokemonTables.put(generation, tableElement);
      }
    }
    
    return pokemonTables;
  }
  
  /**
   * Correct a Pokemon name so that it can be used as part of the URL 
   * @param name
   * @return
   */
  private String correctPokemonName(String name)
  {
    return (name.contains(" ")) ? name.replace(' ', '_') : name;
  }

  /**
   * Scrape a single Pokemon. It may scrape multiple Pokemons due to 
   * there are some Pokemons that have different forms.
   * @param html
   * @param generation
   * @param pokedex
   * @param defaultName
   * @return
   * @throws IOException
   */
  private List<Pokemon> scrapePokemon(String html, int generation, int pokedex, String defaultName) throws IOException
  {
    m_document = Jsoup.parse(html);
    Comment startContent = findStartContent(m_document);
    Element divElement = (Element)(startContent.nextSibling().nextSibling());
    Element mainTableElement = (Element) (divElement.children().stream().
                                         filter(e -> e.tagName().equals("table")).toArray()[1]);

    List<Pokemon> pokemonList = new ArrayList<Pokemon>();
    HashSet<String> formNames = getFormNames(mainTableElement);
    HashMap<String, List<String>> types = getTypes(mainTableElement, defaultName);
    HashMap<String, List<String>> abilities = getAbilities(mainTableElement, defaultName);
    HashMap<String, HashMap<String, Integer>> stats =  getStats(mainTableElement, defaultName);
    
    for (String formName : formNames)
    {
      // Get the corresponding types, abilities, and stats
      String tempName = types.get(formName) == null ? defaultName : formName;
      List<String> formTypes = new ArrayList<String>(types.get(tempName));
      HashSet<String> formAbilities = getFormAbilities(defaultName, formName, abilities);
      HashMap<String, Integer> formStats = getFormStats(defaultName, formName, stats);
      
      // Mega evolution was introduced in Gen 6. Alolan form was introduced in Gen 7
      // Assign the appropriate generation to each form
      int gen = generation;
      if (formName.contains("Mega"))
      {
        gen = 6;
      }
      else if (formName.contains("Alolan"))
      {
        gen = 7;
        List<String> hiddenAbilities = abilities.get(formName + " Hidden Ability");
        if (hiddenAbilities != null) formAbilities.addAll(hiddenAbilities);
      }
      else
      {
        if (abilities.containsKey("Hidden Ability"))
        {
          formAbilities.addAll(abilities.get("Hidden Ability"));
        }
        else if (abilities.containsKey(formName + " Hidden Ability"))
        {
          formAbilities.addAll(abilities.get(formName + " Hidden Ability"));
        }
      }
      
      // Scrape other attribute values
      String category = getCategory(mainTableElement);
      String[] genders = getGenderRatios(mainTableElement);
      String height = getHeight(mainTableElement);
      String weight = getWeight(mainTableElement);
      HashSet<String> eggGroups = getEggGroups(mainTableElement);
      
      // Store all the Pokemon attribute values
      Pokemon pokemon = new Pokemon();
      pokemon.setGeneration(gen);
      pokemon.setPokedex(pokedex);
      pokemon.setName(formName);
      pokemon.setCategory(category);
      pokemon.setAbilities(formAbilities);
      pokemon.setTypes(formTypes);
      pokemon.setMaleRatio(genders[0]);
      pokemon.setFemaleRatio(genders[1]);
      pokemon.setHeight(height);
      pokemon.setWeight(weight);
      pokemon.setEggGroups(eggGroups);
      pokemon.setHp(formStats.get("hp"));
      pokemon.setAttack(formStats.get("attack"));
      pokemon.setDefense(formStats.get("defense"));
      pokemon.setSpAttack(formStats.get("sp.atk"));
      pokemon.setSpDefense(formStats.get("sp.def"));
      pokemon.setSpeed(formStats.get("speed"));
      
      pokemonList.add(pokemon);
    }
    
    return pokemonList;
  }
  
  /**
   * Get all form names of a Pokemon. Such as "Charizard", "Mega Charizard X", etc...
   * @param mainTableElement
   * @return
   */
  private HashSet<String> getFormNames(Element mainTableElement)
  {
    HashSet<String> formNames = new HashSet<String>();
    Element tableRows = mainTableElement.getElementsByTag("table").get(1).child(0).child(1);
    Element tablePictures = tableRows.getElementsByTag("table").get(0);
    
    for (Element element : tablePictures.getElementsByTag("tr"))
    {
      if (!element.hasAttr("style") || !element.attr("style").replaceAll(" ", "").equals("display:none;"))
      {
        Elements imgElements = element.getElementsByTag("img");
        
        for (Element imgElement : imgElements)
        {
          String formName = imgElement.attr("alt");
          formNames.add(formName);
        }
      }
    }
    return formNames;
  }

  /**
   * Get the Pokemon category
   * @param mainTableElement
   * @return
   */
  private String getCategory(Element mainTableElement)
  {
    return mainTableElement.getElementsByAttributeValue("href", "/wiki/Pok%C3%A9mon_category").get(0).text();
  }
  
  /**
   * Get all Pokemon abilities and its respective form's abilities
   * @param mainTableElement
   * @param name
   * @return
   */
  private HashMap<String, List<String>> getAbilities(Element mainTableElement, String name)
  {
    Element element = mainTableElement.getElementsByAttributeValue("title", "Ability").get(0);
    Element parent = element.parent();
    Element tableBodyElement = parent.nextElementSibling().child(0);
    
    HashMap<String, List<String>> abilities = new HashMap<>();
    
    for (Element rowElement : tableBodyElement.children())
    {
      for (Element cellElement : rowElement.children())
      {
        // Avoid unofficial abilities
        if (!cellElement.hasAttr("style") || 
            !cellElement.attr("style").replaceAll(" ", "").contains("display:none"))
        {
          List<String> formAbilities = cellElement.getElementsByTag("a").stream().
                                       map(e -> e.text()).collect(Collectors.toList());
          Element small = (cellElement.getElementsByTag("small").size() == 0) ? 
                          null : cellElement.getElementsByTag("small").get(0);
          String formName = (small == null) ? name : small.text();
          
          if (abilities.containsKey(formName))
          {
            abilities.get(formName).addAll(formAbilities);
          }
          else
          {
            abilities.put(formName, formAbilities);
          }
        }
      }
    }
    
    return abilities;
  }
  
  /**
   * Get all Pokemon types and its respective form's types
   * @param mainTableElement
   * @param name
   * @return
   */
  private HashMap<String, List<String>> getTypes(Element mainTableElement, String name)
  {
    Element element = mainTableElement.getElementsByAttributeValue("title", "Type").get(0);
    Element tableBodyElement = element.parent().nextElementSibling().child(0);
    
    HashMap<String, List<String>> types = new HashMap<>();
    
    for (Element rowElement : tableBodyElement.children())
    {
      for (Element cellElement : rowElement.children())
      {
        // Avoid unofficial types
        if (!cellElement.hasAttr("style") || 
            !cellElement.attr("style").replaceAll(" ", "").contains("display:none"))
        {
          List<String> formTypes = cellElement.getElementsByTag("a").stream().
                                   map(e -> e.text()).collect(Collectors.toList());
          Element small = (cellElement.getElementsByTag("small").size() == 0) ? 
                          null : cellElement.getElementsByTag("small").get(0);
          String formName = (small == null) ? name : small.text();
          
          types.put(formName, formTypes);
        }
      }
    }
    
    return types;
  }
  
  /**
   * Get a Pokemon gender ratio.
   * @param mainTableElement
   * @return
   */
  private String[] getGenderRatios(Element mainTableElement)
  {
    Elements elements = mainTableElement.getElementsByAttributeValue("href", "/wiki/List_of_Pok%C3%A9mon_by_gender_ratio");
    Element tableElement = elements.get(0).parent().nextElementSibling();
    Element gendersElement = tableElement.getElementsByTag("tr").get(1).child(0);
    String male = "0%";
    String female = "0%";
    
    for (Element genderElement : gendersElement.getElementsByTag("span"))
    {
      String[] gender = genderElement.text().trim().toLowerCase().split(" ");
      if (gender.length == 2)
      {
        if (gender[1].equals("male"))
        {
          male = gender[0];
        }
        else if (gender[1].equals("female"))
        {
          female = gender[0];
        }
      }
    }
    return new String[] {male, female};
  }
  
  /**
   * Get a Pokemon's height value in meters
   * @param mainTableElement
   * @return
   */
  private String getHeight(Element mainTableElement)
  {
    Elements elements = mainTableElement.getElementsByAttributeValue("href", "/wiki/List_of_Pok%C3%A9mon_by_height");
    Element tableElement = elements.get(0).parent().nextElementSibling();
    return tableElement.getElementsByTag("tr").get(0).child(1).text();
  }
  
  /**
   * Get a Pokemon's weight value in kg
   * @param mainTableElement
   * @return
   */
  private String getWeight(Element mainTableElement)
  {
    Elements elements = mainTableElement.getElementsByAttributeValue("href", "/wiki/List_of_Pok%C3%A9mon_by_weight");
    Element tableElement = elements.get(0).parent().nextElementSibling();
    return tableElement.getElementsByTag("tr").get(0).child(1).text();
  }
  
  /**
   * Get all Pokemon's egg groups
   * @param mainTableElement
   * @return
   */
  private HashSet<String> getEggGroups(Element mainTableElement)
  {
    HashSet<String> eggGroups = new HashSet<String>();
    Element eggGroupTableElement = mainTableElement.getElementsByAttributeValue("href", "/wiki/Egg_Group").
                                   get(0).parent().nextElementSibling();
    for (Element eggGroupElement : eggGroupTableElement.getElementsByTag("a"))
    {
      eggGroups.add(eggGroupElement.text());
    }
    
    return eggGroups;
  }
  
  /**
   * Get all Pokemon's stats and its respective form's stats
   * @param mainttableElement
   * @param defaultName
   * @return
   */
  private HashMap<String, HashMap<String, Integer>> getStats(Element mainttableElement, String defaultName)
  {
    HashMap<String, HashMap<String, Integer>> allStats = new HashMap<>();
    Element baseStatsElement = null;
    Element statsElement = m_document.getElementById("Base_stats");
    
    // Some pages use Stats while others use Base_stats
    if (statsElement == null)
    {
      baseStatsElement = m_document.getElementById("Stats").parent();
    }
    else
    {
      baseStatsElement = statsElement.parent();
    }
    
    // Get all the stat tables
    Element typeEffectElement = m_document.getElementById("Type_effectiveness").parent();
    Element nextElement = baseStatsElement.nextElementSibling();
    HashMap<String, Integer> latestStats = null;
    
    // Keep finding stat table until we hit the next section, which is type effectiveness
    while (nextElement.siblingIndex() < typeEffectElement.siblingIndex())
    {
      if (nextElement.tagName().equals("table") && nextElement.child(0).children().size() == 10)
      {
        Element prevElement = nextElement.previousElementSibling();
        boolean hasBaseStats = prevElement.text().equals("Base stats") || prevElement.text().equals("Stats");
        String formName = hasBaseStats ? defaultName : prevElement.text();
        HashMap<String, Integer> stats = new HashMap<>();
        
        Elements statsTableElements = nextElement.getElementsByTag("table");
        for (int i = 1; i < statsTableElements.size() - 1; i++)
        {
          String[] stat = statsTableElements.get(i).text().split(":");
          stats.put(stat[0].trim().toLowerCase(), Integer.parseInt(stat[1].trim()));
        }
        allStats.put(formName, stats);
        latestStats = stats;
      }
      nextElement = nextElement.nextElementSibling();
    }
    
    // Add the latest stats table for Pokemon that have their stats change across generations
    allStats.put("latest", latestStats);
    
    return allStats;
  }
   
  /**
   * Get abilities for a particular Pokemon's form (original form included)
   * @param defaultName
   * @param formName
   * @param abilities
   * @return
   */
  private HashSet<String> getFormAbilities(String defaultName, String formName, HashMap<String, List<String>> abilities)
  {
    // Certain Pokemon have different forms but the same ability.
    HashSet<String> formAbilities = null;
    if (abilities.get(formName) == null)
    {
      if (abilities.get(defaultName) != null)
      {
        formAbilities = new HashSet<>(abilities.get(defaultName));
      }
      else
      {
        formAbilities = new HashSet<>(abilities.values().stream().
                                      flatMap(Collection::stream).collect(Collectors.toList()));
      }
    }
    else
    {
      formAbilities = new HashSet<>(abilities.get(formName));
    }
    return formAbilities;
  }

  /**
   * Get stats for a particular Pokemon's form (original form included)
   * @param defaultName
   * @param formName
   * @param stats
   * @return
   */
  private HashMap<String, Integer> getFormStats(String defaultName, String formName, 
                                                HashMap<String, HashMap<String, Integer>> stats)
  {
    // Get the corresponding stats
    String tempName = formName;
    if (stats.get(formName) == null)
    {
      tempName = (stats.get(defaultName) == null) ? "latest" : defaultName;  
    }
    return stats.get(tempName);
  }
  
  /**
   * Find the start of the content in a Pokemon page
   * @param node
   * @return
   */
  private Comment findStartContent(Node node)
  {
    Comment result = null;
    for (int i = 0; i < node.childNodeSize(); i++)
    {
      Node childNode = node.childNode(i);
      if (childNode.nodeName().equals("#comment"))
      {
        Comment comment = (Comment)childNode;
        if (comment.getData().trim().equals("start content"))
        {
          result = comment;
          break;
        }
      }
      else
      {
        result = findStartContent(childNode);
        if (result != null)
        {
          break;
        }
      }
    }
    
    return result;
  }
  
  /**
   * Creat the output file and write all the columns to it
   * @throws IOException
   */
  private void createOutputFile() throws IOException
  {
    openFile(false);
    m_fileWriter.write("Generation\tPokedex#\tName\tCategory\tTypes\tAbilities\tMaleRatio\tFemaleRatio\t" + 
                       "Height\tWeight\tEggGroups\tHP\tAttack\tDefense\tSp.Attack\tSp.Defense\tSpeed\tMovesets\n");
    closeFile();
  }
  
  /**
   * Open the file in write and append mode if requested
   * @param append
   * @throws IOException
   */
  private void openFile(boolean append) throws IOException
  {
    if (m_fileWriter == null)
    {
      File file = new File(OutputFile);
      if (!file.exists())
      {
        file.getParentFile().mkdirs();
      }
      
      m_fileWriter = new FileWriter(file, append);
    }
    else
    {
      closeFile();
      throw new IllegalStateException("File writer is already opened. Closing current file writer.");
    }
  }
  
  /**
   * Close the output file
   */
  private void closeFile()
  {
    if (m_fileWriter != null)
    {
      try
      {
        m_fileWriter.close();
      } 
      catch (IOException ex)
      {}
      m_fileWriter = null;
    }
  }

  /**
  * Given a string URL returns a string with the page contents
  * Adapted from example in 
  * http://docs.oracle.com/javase/tutorial/networking/urls/readingWriting.html
  * @param link
  * @return
  * @throws IOException
  */
  public String getPageFromUrl(String link) throws IOException
  {
    URL thePage = new URL(link);
    URLConnection yc = thePage.openConnection();
    yc.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) " + 
                                        "Chrome/23.0.1271.95 Safari/537.11");
    BufferedReader in = new BufferedReader(new InputStreamReader(yc.getInputStream()));
    
    String inputLine;
    String outputLine = "";
    while ((inputLine = in.readLine()) != null)
    {
      outputLine += inputLine + "\n";
    }
    
    in.close();
    return outputLine;
  }
}
