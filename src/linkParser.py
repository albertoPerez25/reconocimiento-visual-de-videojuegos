import flujo
import os

OUTPUT_DIRECTORY = "dataset/"

class videoFile():
    def __init__(self,path):
        self.flujo = flujo.Flujo(path)

    def stripList(self,list):
        return "".join(cadena for sublista in list for cadena in sublista)

    def makeDir(self,dir):
        try:
            os.mkdir(dir)
            print(f"Directory '{dir}' created successfully.")
        except FileExistsError:
            print(f"Directory '{dir}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{dir}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def getVideoLinks(self):
        char = None
        videoLinks: dict[str, list[str]] = {}

        while(char != 'EOF' and char != False):
            title = None
            links = []
            char = None
            while(char not in ['+','-','/']):
                test = self.flujo.SiguienteLinea()
                char = self.flujo.NewCar()
                self.flujo.Devolver()
            
            if (char == "/" and self.flujo.NewCar() == "/"):
                title = ''.join(self.flujo.SiguienteLinea()).rstrip('\n')
                char = self.flujo.NewCar()
                videoLinks[title] = []
                self.makeDir(OUTPUT_DIRECTORY+title)
            
            while (char in ['+','-']):
                if (char == '+'):
                    links.append(''.join(self.flujo.SiguienteLinea()))
                    char = self.flujo.NewCar()
                elif (char == '-'):
                    links.append(''.join(self.flujo.SiguienteLinea()))
                    char = self.flujo.NewCar()
                
            videoLinks[title].extend(links)
            
            with open(f"{OUTPUT_DIRECTORY}{title}/{title}.txt", 'w') as linksFile:
                clean_links = self.stripList(videoLinks[title])
                linksFile.write(clean_links)
            self.flujo.Devolver()

        return videoLinks
         
if __name__ == "__main__":
    LinkParser = videoFile("./videos_primera_it.txt")

    print(LinkParser.getVideoLinks())

    exit(0)