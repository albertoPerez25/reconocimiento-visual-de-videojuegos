import os

class Opcion:
    """
    Representa una opción dentro de un menú.
    Puede ser una acción (función) o un submenú (instancia de Menu).
    """
    def __init__(self, etiqueta, accion):
        self.etiqueta = etiqueta
        self.accion = accion

    def ejecutar(self):
        # Si la acción es otro menú, iniciamos su bucle
        if isinstance(self.accion, Menu):
            self.accion.iniciar()
        # Si la acción es una función ejecutable (callable), la llamamos
        elif callable(self.accion):
            self.accion()
            input("\nPresiona ENTER para continuar...")
        else:
            print(f"Error: La acción de '{self.etiqueta}' no es válida.")

class Menu:
    """
    Clase principal para crear menús y submenús.
    """
    def __init__(self, titulo, tecla_volver='x'):
        self.titulo = titulo
        self.opciones = []
        self.tecla_volver = tecla_volver

    def agregar_opcion(self, etiqueta, accion):
        """
        Agrega una opción al menú.
        :param etiqueta: Texto que se mostrará.
        :param accion: Puede ser una función o instancia de otro Menu.
        """
        self.opciones.append(Opcion(etiqueta, accion))

    def _limpiar_pantalla(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def _mostrar_interfaz(self):
        self._limpiar_pantalla()
        print(f"=== {self.titulo.upper()} ===")
        print("-" * (len(self.titulo) + 8))
        
        # Listar opciones numeradas
        for i, opcion in enumerate(self.opciones, 1):
            tipo = ">>" if isinstance(opcion.accion, Menu) else "  "
            print(f"{i}. {opcion.etiqueta} {tipo}")
            
        print("-" * (len(self.titulo) + 8))
        print(f"[{self.tecla_volver}] Volver / Salir")

    def iniciar(self):
        while True:
            self._mostrar_interfaz()
            eleccion = input("\n> Selecciona una opción: ").strip().lower()

            if eleccion == self.tecla_volver:
                break

            if eleccion.isdigit():
                indice = int(eleccion) - 1
                if 0 <= indice < len(self.opciones):
                    # Ejecutar la opción seleccionada
                    self.opciones[indice].ejecutar()
                else:
                    input("Número fuera de rango. Escribe uno de los números de la lista. Presiona ENTER.")
            else:
                input("Entrada inválida. Escribe uno de los números de la lista o la tecla para volver. Presiona ENTER.")

# ==========================================
# ZONA DE EJEMPLO DE USO
# ==========================================

# 1. Definimos algunas funciones de acción simples
def saludar():
    print("¡Hola! Has ejecutado una función simple.")

def calcular_suma():
    try:
        a = float(input("Introduce el primer número: "))
        b = float(input("Introduce el segundo número: "))
        print(f"El resultado es: {a + b}")
    except ValueError:
        print("Error: Debes introducir números.")

def mostrar_info_sistema():
    import sys
    print(f"Plataforma: {sys.platform}")
    print(f"Versión de Python: {sys.version}")

# 2. Construimos la estructura de Menús

# --- Submenú de Operaciones ---
menu_dataset = Menu("Generación de dataset")
menu_dataset.agregar_opcion("Descargar vídeos", saludar)
menu_dataset.agregar_opcion("Extraer fotogramas", saludar)

menu_modelo = Menu("Modelos de aprendizaje")
menu_modelo.agregar_opcion("MLP (Básico)", saludar)
menu_modelo.agregar_opcion("CNN (Propia)", saludar)
menu_modelo.agregar_opcion("Transfer Learning", saludar)

menu_entrenamiento = Menu("Herramientas de entrenamiento")
menu_entrenamiento.agregar_opcion("Entrenar modelo", menu_modelo)
menu_entrenamiento.agregar_opcion("Generación de dataset", menu_dataset)

# --- Submenú de Configuración ---
menu_predecir = Menu("Predicción de imágenes")
menu_predecir.agregar_opcion("MLP (Básico)", saludar)
menu_predecir.agregar_opcion("CNN (Propia)", saludar)
menu_predecir.agregar_opcion("Transfer Learning", saludar)

# --- Menú Principal ---
principal = Menu("Menú Principal - ¿Qué deseas hacer?") # Aquí cambiamos la tecla de salida a 'x'
principal.agregar_opcion("Predecir imagen", menu_predecir)
principal.agregar_opcion("Herramientas de entrenamiento", menu_entrenamiento)

# 3. Ejecutamos el programa
if __name__ == "__main__":
    # Arrancamos el menú raíz
    principal.iniciar()
    print("\n¡Hasta luego!")